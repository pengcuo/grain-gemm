# SPDX-License-Identifier: Apache-2.0
"""Pipelined INT8 Tensor Core GEMM with exact integer K-group reductions."""

import triton
import triton.language as tl


@triton.jit
def grouped_gemm(
    A, B, SA, SB, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    SAM: tl.constexpr, SAG: tl.constexpr, SBG: tl.constexpr, SBN: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, GROUP: tl.constexpr,
    STAGES: tl.constexpr, SWIZZLE: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, TILE_M)
    tiles_n = tl.cdiv(N, TILE_N)
    band = pid // (SWIZZLE * tiles_n)
    first_m = band * SWIZZLE
    band_m = tl.minimum(tiles_m - first_m, SWIZZLE)
    tile_m = first_m + pid % band_m
    tile_n = (pid % (SWIZZLE * tiles_n)) // band_m
    rows = tile_m * TILE_M + tl.arange(0, TILE_M)
    cols = tile_n * TILE_N + tl.arange(0, TILE_N)
    offsets = tl.arange(0, GROUP)
    total = tl.full((TILE_M, TILE_N), 0, tl.float32)

    # GROUP is the mathematical quantization group, independent of M/N tiling.
    # Compile-time strides/bounds remove address divisions and full-tile masks.
    # Software pipelining overlaps the next groups' copies with this group's MMA.
    for group in tl.range(tl.cdiv(K, GROUP), num_stages=STAGES):
        ks = group * GROUP + offsets
        a = tl.load(
            A + rows[:, None] * AM + ks[None, :] * AK,
            (rows[:, None] < M) & (ks[None, :] < K), other=0,
        )
        b = tl.load(
            B + ks[:, None] * BK + cols[None, :] * BN,
            (ks[:, None] < K) & (cols[None, :] < N), other=0,
        )
        sa = tl.load(SA + rows * SAM + group * SAG, rows < M, other=0)
        sb = tl.load(SB + group * SBG + cols * SBN, cols < N, other=0)
        partial = tl.dot(a, b, out_dtype=tl.int32)
        total = tl.fma(partial.to(tl.float32), sa[:, None] * sb[None, :], total)

    tl.store(
        C + rows[:, None] * N + cols[None, :], total,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def launch(a, b, scale_a, scale_b, out, group_size, config):
    m, k = a.shape
    n = b.shape[1]
    bm, bn = config["block_m"], config["block_n"]
    return grouped_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, scale_a, scale_b, out, m, n, k,
        *a.stride(), *b.stride(), *scale_a.stride(), *scale_b.stride(),
        bm, bn, group_size, config["num_stages"], config["swizzle"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
