# SPDX-License-Identifier: Apache-2.0
"""Validated inference API and architecture-aware kernel dispatch."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from .kernels.triton import launch


def _validate(a, b, scale_a, scale_b, group_size, output_dtype):
    if type(group_size) is not int or group_size not in (32, 64, 128, 256):
        raise ValueError("group_size must be one of 32, 64, 128, or 256")
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("output_dtype must be torch.float32, torch.float16, or torch.bfloat16")
    tensors = (("a", a), ("b", b), ("scale_a", scale_a), ("scale_b", scale_b))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional strided tensor")
        expected = torch.int8 if name in ("a", "b") else torch.float32
        if tensor.dtype != expected:
            raise TypeError(f"{name} must have dtype {expected}")
    m, k = a.shape
    n = b.shape[1]
    if b.shape[0] != k:
        raise ValueError("a.shape[1] must equal b.shape[0] (the K dimension)")
    groups = (k + group_size - 1) // group_size
    for name, tensor, shape in (
        ("scale_a", scale_a, (m, groups)), ("scale_b", scale_b, (groups, n))
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    for name, tensor in tensors:
        if tensor.device != a.device:
            raise ValueError(f"{name} must be on the same device as a ({a.device})")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.numel() and any(s <= 0 for s in tensor.stride()):
            raise ValueError(f"{name} must have positive strides")
        if tensor.numel() and sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride())) > 2**31 - 1:
            raise ValueError(f"{name} exceeds the supported 32-bit relative element offsets")
    if m * n > 2**31:
        raise ValueError("output exceeds the supported 32-bit relative element offsets")
    if torch.cuda.get_device_capability(a.device)[0] < 8:
        raise ValueError("GrainGEMM requires an NVIDIA SM80 or newer GPU")
    return m, n, k


@lru_cache(maxsize=1)
def _dispatch_table():
    path = Path(__file__).with_name("kernels") / "configs" / "sm121_g256.json"
    return json.loads(path.read_text()) if path.exists() else {"square_configs": {}}


def _native_compatible(a, b, scale_a, scale_b, group_size, output_dtype):
    m, k = a.shape
    n = b.shape[1]
    return (group_size == 256 and output_dtype == torch.bfloat16
            and m > 0 and n > 0 and k > 0
            and m % 64 == n % 64 == k % 256 == 0
            and a.stride() == (k, 1) and b.stride() == (1, k)
            and a.data_ptr() % 16 == b.data_ptr() % 16 == 0
            and scale_a is not None and scale_b is not None
            and scale_a.is_contiguous() and scale_b.is_contiguous())


def get_kernel_config(a, b, *, group_size=256, output_dtype=torch.float32,
                      scale_a=None, scale_b=None, backend="auto"):
    """Describe the selected kernel without compiling or benchmarking it.

    Pass the same dtype and scale tensors as int8_gemm for an exact dispatch
    description. Omitted scales disable selection of the native fast path.
    The measured table is specific to SM121/G256/BF16 and the documented layout;
    other architectures and shapes retain a portable Triton implementation.
    """
    if backend not in ("auto", "triton", "cuda"):
        raise ValueError("backend must be 'auto', 'triton', or 'cuda'")
    capability = torch.cuda.get_device_capability(a.device)
    m, k = a.shape
    n = b.shape[1]
    aligned = (a.stride() == (k, 1) and b.stride() == (1, k)
               and k % 256 == 0 and min(m, n) >= 64)
    table = _dispatch_table()
    entry = (table["square_configs"].get(str(m))
             if capability == (12, 1) and m == n == k and group_size == 256
             and aligned and output_dtype == torch.bfloat16 else None)
    if capability == (12, 1) and group_size == 256 and aligned:
        config = dict(backend="triton", block_m=64, block_n=128,
                      num_warps=8, num_stages=3, swizzle=8)
        policy = "sm121_g256_heuristic"
    else:
        config = dict(backend="triton", block_m=32, block_n=64,
                      num_warps=4, num_stages=2, swizzle=8)
        policy = "portable"
    if entry:
        config = dict(entry["triton"])
        policy = "sm121_g256_measured"
    wants_native = backend == "cuda" or (backend == "auto" and entry
                                        and entry["selected"]["backend"] == "cuda")
    if wants_native:
        from .kernels import cuda
        supported = (capability == (12, 1) and _native_compatible(
            a, b, scale_a, scale_b, group_size, output_dtype))
        if backend == "cuda" and not supported:
            raise ValueError("cuda requires SM121, G256, BF16 output, M/N multiples of 64, "
                             "K a positive multiple of 256, aligned contiguous A and B.T, "
                             "and contiguous FP32 scales")
        if supported and cuda.is_available():
            config_id = (entry.get("cuda", entry["selected"]).get("config_id", 7)
                         if entry else (7 if m % 128 == n % 128 == 0 else 0))
            config = dict(backend="cuda", config_id=config_id)
            policy = "sm121_g256_measured" if entry else "sm121_g256_explicit"
        elif backend == "cuda":
            raise RuntimeError("CUDA kernel is not built; run tools/build_cuda.py --cutlass-dir PATH")
    return dict(policy=policy, compute_capability=list(capability), **config)


def int8_gemm(a, b, scale_a, scale_b, *, group_size=256,
              output_dtype=torch.float32, backend="auto"):
    """Compute INT8 A[M,K] @ B[K,N] with independent row/column K-group scales.

    FP32 scale shapes are [M, ceil(K/G)] and [ceil(K/G), N]. Each G-element dot
    product accumulates in INT32, then contributes to an FP32 scaled sum. The
    result is cast once to FP32 (default), FP16 or BF16. G can be 32/64/128/256.
    All inputs must share an SM80+ CUDA device; positive-stride views and tails
    are supported without packing. Quantization and autograd are not included.

    backend='auto' selects settings by architecture, shape and layout;
    backend='triton' explicitly selects Triton; backend='cuda' requires the optional
    native CuTe build and its supported layout. Auto falls back to Triton when
    the native build or layout is unavailable. Warm up the
    function before capturing it in a CUDA graph. No tuning runs occur inside
    this call.
    """
    if backend not in ("auto", "triton", "cuda"):
        raise ValueError("backend must be 'auto', 'triton', or 'cuda'")
    m, n, k = _validate(a, b, scale_a, scale_b, group_size, output_dtype)
    with torch.cuda.device(a.device):
        if m == 0 or n == 0 or k == 0:
            return torch.zeros((m, n), device=a.device, dtype=output_dtype)
        result = torch.empty((m, n), device=a.device, dtype=output_dtype)
        config = get_kernel_config(
            a, b, group_size=group_size, output_dtype=output_dtype,
            scale_a=scale_a, scale_b=scale_b, backend=backend,
        )
        if config["backend"] == "cuda":
            from .kernels import cuda
            cuda.launch(a, b, scale_a, scale_b, result, config["config_id"])
        else:
            launch(a, b, scale_a, scale_b, result, group_size, config)
    return result
