# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2024 SGLang Team
# Extracted from SGLang, licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
# Upstream commit: 8ac19cc19f8ade51ed203478f17c9a09c706d730
# Source: python/sglang/kernels/ops/quantization/int8_kernel.py
# https://github.com/sgl-project/sglang/blob/8ac19cc19f8ade51ed203478f17c9a09c706d730/python/sglang/kernels/ops/quantization/int8_kernel.py
# Modified for GrainGEMM: extracted the kernel without SGLang dependencies;
# replaced the wrapper with the API and explicit launch configuration below.
# The decorated _w8a8_block_int8_matmul function is unchanged from upstream.

"""Standalone SGLang Triton baseline for row/column K-group INT8 GEMM.

This is a configurable baseline, with no claim of an optimal launch configuration.
It consumes already quantized tensors; quantization and autograd are not provided.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_block_int8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sglang_int8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    group_size: int = 256,
    output_dtype: torch.dtype = torch.float32,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Multiply INT8 matrices with one scale per row/column and K-group.

    Args:
        a: INT8 matrix with logical shape ``[M, K]``.
        b: INT8 matrix with logical shape ``[K, N]``.
        scale_a: FP32 scales with shape ``[M, ceil(K / group_size)]``.
        scale_b: FP32 scales with shape ``[ceil(K / group_size), N]``.
        group_size: Number of K elements per quantization group: 32/64/128/256.
        output_dtype: FP32 (default), FP16, or BF16.
        block_m: Power-of-two output tile height, at least 16.
        block_n: Power-of-two output tile width, at least 16. This is distinct
            from weight scale granularity, which is always one output column.
        num_warps: Number of warps per program: 1, 2, 4, or 8.
        num_stages: Positive software-pipeline stage count.

    All tensors must be on the same CUDA device. Positive-stride transposed or
    sliced views are accepted without copies. The last K-group may be partial;
    out-of-range elements are masked. Each full/partial group is accumulated in
    INT32, converted and scaled in FP32, then added to the FP32 running sum.
    The result is cast once to ``output_dtype``. This inference-only baseline
    does not implement autograd.

    Launch parameters are explicit and are not autotuned. Configurations that
    exceed the target GPU's resource limits can fail at compilation or launch.
    """
    if type(group_size) is not int or group_size not in (32, 64, 128, 256):
        raise ValueError("group_size must be one of 32, 64, 128, or 256")
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            "output_dtype must be torch.float32, torch.float16, or torch.bfloat16"
        )
    for name, value in (("block_m", block_m), ("block_n", block_n)):
        if type(value) is not int or value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two and at least 16")
    if type(num_warps) is not int or num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if type(num_stages) is not int or num_stages < 1:
        raise ValueError("num_stages must be a positive integer")

    tensors = (("a", a), ("b", b), ("scale_a", scale_a), ("scale_b", scale_b))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional strided tensor")
    for name, tensor in tensors[:2]:
        if tensor.dtype != torch.int8:
            raise TypeError(f"{name} must have dtype torch.int8")
    for name, tensor in tensors[2:]:
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError("a.shape[1] must equal b.shape[0] (the K dimension)")
    n = b.shape[1]
    groups = triton.cdiv(k, group_size)
    if scale_a.shape != (m, groups):
        raise ValueError(
            f"scale_a must have shape {(m, groups)}, got {tuple(scale_a.shape)}"
        )
    if scale_b.shape != (groups, n):
        raise ValueError(
            f"scale_b must have shape {(groups, n)}, got {tuple(scale_b.shape)}"
        )
    for name, tensor in tensors:
        if tensor.device != a.device:
            raise ValueError(f"{name} must be on the same device as a ({a.device})")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.numel() and any(stride <= 0 for stride in tensor.stride()):
            raise ValueError(f"{name} must have positive strides")

    # This wrapper intentionally does not reuse SGLang's config loader: its
    # default couples TILE_N to group_n, yielding TILE_N=1 for per-column
    # weight scales. Keep output tiling independent of scale granularity.
    with torch.cuda.device(a.device):
        if m == 0 or n == 0 or k == 0:
            return torch.zeros((m, n), device=a.device, dtype=output_dtype)
        result = torch.empty((m, n), device=a.device, dtype=output_dtype)
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _w8a8_block_int8_matmul[grid](
            a,
            b,
            result,
            scale_a,
            scale_b,
            m,
            n,
            k,
            1,  # group_n: independent scale for every output column.
            group_size,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            result.stride(0),
            result.stride(1),
            scale_a.stride(0),
            scale_a.stride(1),
            scale_b.stride(0),
            scale_b.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=group_size,
            GROUP_SIZE_M=32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return result
