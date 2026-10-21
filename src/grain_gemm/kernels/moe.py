# SPDX-License-Identifier: Apache-2.0
"""Device-side routing, quantization, and ragged INT8 MoE building blocks.

The public FFN API validates the inputs.  These helpers keep routing data on the
GPU, including when a captured graph is replayed with a different assignment.
No helper launches one GEMM per expert or reads routing counts on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@dataclass(frozen=True)
class Routes:
    """A fixed-size routing workspace; valid assignments occupy a sorted prefix.

    ``order[slot]`` is the original flattened [token, top-k] assignment.
    ``inverse[assignment]`` is its sorted slot, or -1 for an invalid/masked row.
    The final offset is the number of valid assignments, kept on the device.
    ``statistics`` contains the unmasked-invalid-id count, largest expert count,
    and total valid assignment count, in that order. Tile metadata is optional
    for native execution and can be added with ``prepare_ragged_tiles``.
    """

    order: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    tile_offsets: torch.Tensor | None
    inverse: torch.Tensor
    effective_ids: torch.Tensor
    tile_experts: torch.Tensor | None
    tile_starts: torch.Tensor | None
    tokens: int
    top_k: int
    experts: int
    block_m: int
    max_tiles: int
    statistics: torch.Tensor


@triton.jit
def _routing_keys(IDS, MASK, KEYS, INVALID_COUNTS, S: tl.constexpr, R: tl.constexpr,
                  E: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
    assignments = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + assignments, assignments < S, other=E)
    enabled = assignments < S
    if HAS_MASK:
        enabled = enabled & tl.load(
            MASK + assignments // R, assignments < S, other=False)
    # Keep INT64 ids at their original width until after the range check:
    # narrowing an invalid id first could wrap it into a valid expert index.
    in_range = (ids >= 0) & (ids < E)
    valid = enabled & in_range
    tl.store(KEYS + assignments, tl.where(valid, ids, E), assignments < S)
    invalid_count = tl.sum((enabled & ~in_range).to(tl.int32), axis=0)
    tl.store(INVALID_COUNTS + tl.program_id(0), invalid_count)


@triton.jit
def _routing_histogram_inverse(KEYS, ORDER, COUNTS, INVERSE,
                               S: tl.constexpr, E: tl.constexpr,
                               BLOCK: tl.constexpr):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(ORDER + slots, slots < S, other=0)
    expert = tl.load(KEYS + assignment, slots < S, other=E)
    valid = (slots < S) & (expert >= 0) & (expert < E)
    tl.atomic_add(COUNTS + expert, 1, valid, sem="relaxed")
    tl.store(INVERSE + assignment, tl.where(valid, slots, -1), slots < S)


@triton.jit
def _routing_prefix(COUNTS, OFFSETS, TILE_OFFSETS, INVALID_COUNTS, STATISTICS,
                    E: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr,
                    KEY_BLOCKS: tl.constexpr, INVALID_BLOCK: tl.constexpr,
                    INCLUDE_TILES: tl.constexpr):
    experts = tl.arange(0, BLOCK)
    counts = tl.load(COUNTS + experts, experts < E, other=0)
    prefix = tl.cumsum(counts)
    tl.store(OFFSETS, 0)
    tl.store(OFFSETS + experts + 1, prefix, experts < E)
    if INCLUDE_TILES:
        tile_prefix = tl.cumsum(tl.cdiv(counts, BM))
        tl.store(TILE_OFFSETS, 0)
        tl.store(TILE_OFFSETS + experts + 1, tile_prefix, experts < E)
    invalid_count = tl.full((), 0, tl.int32)
    if KEY_BLOCKS > 0:
        blocks = tl.arange(0, INVALID_BLOCK)
        invalid = tl.load(INVALID_COUNTS + blocks, blocks < KEY_BLOCKS, other=0)
        invalid_count = tl.sum(invalid, axis=0)
    tl.store(STATISTICS, invalid_count)
    tl.store(STATISTICS + 1, tl.max(counts, axis=0))
    tl.store(STATISTICS + 2, tl.sum(counts, axis=0))


@triton.jit
def _routing_chunk_histogram(IDS, MASK, KEYS, HISTOGRAM, INVALID_COUNTS,
                             S: tl.constexpr, R: tl.constexpr, E: tl.constexpr,
                             HAS_MASK: tl.constexpr, BINS: tl.constexpr,
                             BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    assignments = chunk * BLOCK + tl.arange(0, BLOCK)
    present = assignments < S
    ids = tl.load(IDS + assignments, present, other=E)
    enabled = present
    if HAS_MASK:
        enabled = enabled & tl.load(MASK + assignments // R, present, other=False)
    # Check at the original index width before narrowing into histogram bins.
    in_range = (ids >= 0) & (ids < E)
    keys = tl.where(enabled & in_range, ids, E)
    tl.store(KEYS + assignments, keys, present)
    histogram = tl.histogram(keys.to(tl.int32), BINS, mask=present)
    bins = tl.arange(0, BINS)
    tl.store(HISTOGRAM + chunk * BINS + bins, histogram)
    tl.store(INVALID_COUNTS + chunk,
             tl.sum((enabled & ~in_range).to(tl.int32), axis=0))


@triton.jit
def _routing_chunk_prefix(HISTOGRAM, CHUNK_BASES, COUNTS, OFFSETS,
                          TILE_OFFSETS, INVALID_COUNTS, STATISTICS,
                          E: tl.constexpr, CHUNKS: tl.constexpr,
                          CHUNK_BLOCK: tl.constexpr, BINS: tl.constexpr,
                          BM: tl.constexpr, INCLUDE_TILES: tl.constexpr):
    chunks = tl.arange(0, CHUNK_BLOCK)
    experts = tl.arange(0, BINS)
    histogram = tl.load(HISTOGRAM + chunks[:, None] * BINS + experts[None, :],
                        chunks[:, None] < CHUNKS, other=0)
    totals = tl.sum(histogram, axis=0)
    expert_ends = tl.cumsum(totals, axis=0)
    # The E-th bin contains masked/invalid assignments and forms the sorted
    # tail. Including it keeps the entire ORDER stable, not only valid rows.
    expert_starts = expert_ends - totals
    bases = (tl.cumsum(histogram, axis=0) - histogram
             + expert_starts[None, :])
    tl.store(CHUNK_BASES + chunks[:, None] * BINS + experts[None, :],
             bases, chunks[:, None] < CHUNKS)
    tl.store(COUNTS + experts, totals, experts < E)
    tl.store(OFFSETS, 0)
    tl.store(OFFSETS + experts + 1, expert_ends, experts < E)
    valid_counts = tl.where(experts < E, totals, 0)
    if INCLUDE_TILES:
        tl.store(TILE_OFFSETS, 0)
        tl.store(TILE_OFFSETS + experts + 1,
                 tl.cumsum(tl.cdiv(valid_counts, BM), axis=0), experts < E)
    invalid = tl.load(INVALID_COUNTS + chunks, chunks < CHUNKS, other=0)
    tl.store(STATISTICS, tl.sum(invalid, axis=0))
    tl.store(STATISTICS + 1, tl.max(valid_counts, axis=0))
    tl.store(STATISTICS + 2, tl.sum(valid_counts, axis=0))


@triton.jit
def _routing_chunk_scatter(KEYS, CHUNK_BASES, ORDER, INVERSE,
                           S: tl.constexpr, E: tl.constexpr,
                           BINS: tl.constexpr, BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    expert = tl.program_id(1)
    assignments = chunk * BLOCK + tl.arange(0, BLOCK)
    keys = tl.load(KEYS + assignments, assignments < S, other=-1)
    selected = (assignments < S) & (keys == expert)
    local_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    base = tl.load(CHUNK_BASES + chunk * BINS + expert)
    slots = base + local_rank
    tl.store(ORDER + slots, assignments, selected)
    tl.store(INVERSE + assignments, tl.where(expert < E, slots, -1), selected)


@triton.jit
def _routing_tile_prefix(COUNTS, TILE_OFFSETS, E: tl.constexpr,
                         BM: tl.constexpr, BLOCK: tl.constexpr):
    experts = tl.arange(0, BLOCK)
    counts = tl.load(COUNTS + experts, experts < E, other=0)
    tile_prefix = tl.cumsum(tl.cdiv(counts, BM))
    tl.store(TILE_OFFSETS, 0)
    tl.store(TILE_OFFSETS + experts + 1, tile_prefix, experts < E)


@triton.jit
def _routing_tiles(OFFSETS, TILE_OFFSETS, TILE_EXPERTS, TILE_STARTS,
                   E: tl.constexpr, MAX_TILES: tl.constexpr, BM: tl.constexpr,
                   STEPS: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total_tiles = tl.load(TILE_OFFSETS + E)
    active = (tiles < MAX_TILES) & (tiles < total_tiles)
    target = tl.where(active, tiles, 0)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), E, tl.int32)
    # Find the first expert whose exclusive tile end is greater than target.
    # Equal offsets for empty experts are skipped naturally.
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        end = tl.load(TILE_OFFSETS + mid + 1, mid < E, other=2147483647)
        right = end <= target
        unfinished = lo < hi
        lo = tl.where(unfinished & right, mid + 1, lo)
        hi = tl.where(unfinished & ~right, mid, hi)
    expert = lo
    first_slot = tl.load(OFFSETS + expert, active & (expert < E), other=0)
    first_tile = tl.load(TILE_OFFSETS + expert, active & (expert < E), other=0)
    starts = first_slot + (target - first_tile) * BM
    tl.store(TILE_EXPERTS + tiles, tl.where(active, expert, -1), tiles < MAX_TILES)
    tl.store(TILE_STARTS + tiles, tl.where(active, starts, 0), tiles < MAX_TILES)


def _routing_chunk_size(slots, num_experts, device):
    """Choose stable-counting chunks without reading device routing values."""
    if not 0 < slots <= 262144 or num_experts > 64:
        return 0
    if torch.cuda.get_device_capability(device) == (11, 0):
        # Larger chunks bound the prefix workspace while retaining counting
        # sort for Thor batches that would otherwise use the general sort.
        return 2048
    return 512 if slots <= 65536 else 0


def prepare_routes(topk_ids, num_experts, token_mask=None, block_m=32,
                   include_tiles=True):
    """Sort assignments and construct fixed-size, GPU-resident expert metadata.

    Invalid ids are guarded as well as masked tokens, so neither can generate
    an out-of-bounds expert access.  The public API may reject invalid ids using
    its own validation policy.  Sorting is stable to preserve token order inside
    an expert; all allocations depend only on tensor shapes, never GPU values.
    ``include_tiles=False`` omits all three ragged tile tensors and their work.
    Statistics and the metadata needed for native packing are always generated.
    """
    tokens, top_k = topk_ids.shape
    slots = tokens * top_k
    effective_ids = torch.empty((slots,), dtype=topk_ids.dtype, device=topk_ids.device)
    # For ordinary expert counts a small counting sort replaces the general
    # Torch stable sort and its temporary allocations/launches. The bounded
    # prefix workspace keeps the fallback practical for very large inputs.
    block = _routing_chunk_size(slots, num_experts, topk_ids.device)
    counts = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    inverse = torch.empty((slots,), dtype=torch.int32, device=topk_ids.device)
    offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tile_offsets = torch.empty_like(offsets) if include_tiles else None
    statistics = torch.empty((3,), dtype=torch.int32, device=topk_ids.device)
    if block:
        chunks = triton.cdiv(slots, block)
        bins = triton.next_power_of_2(num_experts + 1)
        histogram = torch.empty((chunks, bins), dtype=torch.int32, device=topk_ids.device)
        bases = torch.empty_like(histogram)
        invalid_counts = torch.empty((chunks,), dtype=torch.int32, device=topk_ids.device)
        order = torch.empty((slots,), dtype=torch.int64, device=topk_ids.device)
        _routing_chunk_histogram[(chunks,)](
            topk_ids, token_mask, effective_ids, histogram, invalid_counts,
            slots, top_k, num_experts, token_mask is not None, bins, block,
            num_warps=4,
        )
        chunk_block = triton.next_power_of_2(chunks)
        _routing_chunk_prefix[(1,)](
            histogram, bases, counts, offsets, tile_offsets, invalid_counts, statistics,
            num_experts, chunks, chunk_block, bins, block_m, include_tiles,
            num_warps=8 if chunk_block * bins >= 8192 else 4,
        )
        _routing_chunk_scatter[(chunks, num_experts + 1)](
            effective_ids, bases, order, inverse, slots, num_experts, bins, block,
            num_warps=4,
        )
    else:
        key_blocks = triton.cdiv(slots, 1024)
        invalid_counts = torch.empty((key_blocks,), dtype=torch.int32, device=topk_ids.device)
        if slots:
            _routing_keys[(key_blocks,)](
                topk_ids, token_mask, effective_ids, invalid_counts,
                slots, top_k, num_experts, token_mask is not None, 1024,
            )
        order = torch.argsort(effective_ids, stable=True)
        counts.zero_()
        if slots:
            _routing_histogram_inverse[(triton.cdiv(slots, 1024),)](
                effective_ids, order, counts, inverse, slots, num_experts, 1024,
            )
        _routing_prefix[(1,)](
            counts, offsets, tile_offsets, invalid_counts, statistics,
            num_experts, block_m, triton.next_power_of_2(num_experts),
            key_blocks, triton.next_power_of_2(max(1, key_blocks)), include_tiles,
        )
    # sum(ceil(count_e / BM)) <= ceil(S / BM) + E - 1 for S > 0.
    max_tiles = triton.cdiv(slots, block_m) + num_experts
    routes = Routes(order, counts, offsets, tile_offsets, inverse, effective_ids,
                    None, None, tokens, top_k, num_experts,
                    block_m, max_tiles, statistics)
    return prepare_ragged_tiles(routes) if include_tiles else routes


def prepare_ragged_tiles(routes):
    """Return immutable routing metadata with ragged GEMM tile tensors added.

    Native execution can defer this work until an eager auto decision selects
    the ragged path. No routing values are copied to the host, and the original
    ``Routes`` object and its statistics remain unchanged.
    """
    if routes.tile_experts is not None and routes.tile_starts is not None:
        return routes
    tile_offsets = routes.tile_offsets
    if tile_offsets is None:
        tile_offsets = torch.empty_like(routes.offsets)
        _routing_tile_prefix[(1,)](
            routes.counts, tile_offsets, routes.experts, routes.block_m,
            triton.next_power_of_2(routes.experts),
        )
    tile_experts = torch.empty((routes.max_tiles,), dtype=torch.int32,
                               device=routes.counts.device)
    tile_starts = torch.empty_like(tile_experts)
    _routing_tiles[(triton.cdiv(routes.max_tiles, 128),)](
        routes.offsets, tile_offsets, tile_experts, tile_starts, routes.experts,
        routes.max_tiles, routes.block_m, (routes.experts + 1).bit_length(), 128,
    )
    return replace(routes, tile_offsets=tile_offsets,
                   tile_experts=tile_experts, tile_starts=tile_starts)


@triton.jit
def _symmetric_quantize_row_block(values):
    """Independent G256 reductions for several rows in one program."""
    maximum = tl.maximum(tl.max(tl.abs(values), axis=1), 1.0e-12)
    scale = libdevice.div_rn(maximum, 127.0)
    rounded = libdevice.nearbyint(libdevice.div_rn(values, scale[:, None]))
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    return quantized, scale


@triton.jit
def _quantize_row_block(X, Q, S, M: tl.constexpr, K: tl.constexpr,
                        GROUP: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    values = tl.load(X + rows[:, None] * K + cols[None, :],
                     rows[:, None] < M, other=0).to(tl.float32)
    quantized, scales = _symmetric_quantize_row_block(values)
    tl.store(Q + rows[:, None] * K + cols[None, :], quantized, rows[:, None] < M)
    tl.store(S + rows * (K // GROUP) + group, scales, rows < M)


def quantize_rows(x, group_size=256, *, rows_per_program=4):
    """Quantize contiguous floating rows; return INT8 data and FP32 row scales."""
    width = x.shape[-1]
    rows = prod(x.shape[:-1])
    quantized = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((*x.shape[:-1], width // group_size),
                         dtype=torch.float32, device=x.device)
    if rows:
        _quantize_row_block[(triton.cdiv(rows, rows_per_program), width // group_size)](
            x, quantized, scales, rows, width, group_size, rows_per_program, num_warps=4,
            enable_fp_fusion=False,
        )
    return quantized, scales


@triton.jit
def _ragged_gemm(A, B, SA, SB, C, ORDER, OFFSETS, TILE_EXPERTS, TILE_STARTS,
                  N: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
                  SOURCE_IS_TOKEN: tl.constexpr, STAGES: tl.constexpr):
    column_tiles = tl.cdiv(N, BN)
    tile = tl.program_id(0) // column_tiles
    expert = tl.load(TILE_EXPERTS + tile)
    if expert >= 0:
        rows = tl.load(TILE_STARTS + tile) + tl.arange(0, BM)
        row_end = tl.load(OFFSETS + expert + 1)
        row_valid = rows < row_end
        source_rows = rows
        if SOURCE_IS_TOKEN:
            assignments = tl.load(ORDER + rows, row_valid, other=0)
            source_rows = assignments // R
        cols = (tl.program_id(0) % column_tiles) * BN + tl.arange(0, BN)
        kk = tl.arange(0, GROUP)
        total = tl.full((BM, BN), 0, tl.float32)
        for group in tl.range(K // GROUP, num_stages=STAGES):
            ks = group * GROUP + kk
            a = tl.load(A + source_rows[:, None] * K + ks[None, :],
                        row_valid[:, None], other=0)
            # B is the contiguous Linear view [E,N,K].  No weight copy is made.
            b = tl.load(B + expert * N * K + cols[None, :] * K + ks[:, None],
                        cols[None, :] < N, other=0)
            sa = tl.load(SA + source_rows * (K // GROUP) + group,
                         row_valid, other=0)
            sb = tl.load(SB + expert * (K // GROUP) * N + group * N + cols,
                         cols < N, other=0)
            partial = tl.dot(a, b, out_dtype=tl.int32)
            # Match the existing int8_gemm Triton accumulation order and FMA.
            total = tl.fma(partial.to(tl.float32), sa[:, None] * sb[None, :], total)
        tl.store(C + rows[:, None] * N + cols[None, :], total,
                 row_valid[:, None] & (cols[None, :] < N))


def ragged_gemm(q, w_linear, scale_a, scale_b, routes, *, output_dtype,
                source_is_token=False, group_size=256, block_n=64):
    """Run all experts in one ragged launch; output is [sorted slot, N].

    ``w_linear`` is [E,N,K] contiguous, i.e. the zero-copy transpose of the
    public [E,K,N] GEMM view.  Only the valid sorted prefix is written.  Pass
    ``routes`` to swiglu_quantize or combine so invalid tail slots are ignored.
    """
    _, n, k = w_linear.shape
    slots = routes.tokens * routes.top_k
    out = torch.empty((slots, n), dtype=output_dtype, device=q.device)
    if slots:
        # Flatten to avoid CUDA grid.y's 65535 limit for wide expert weights.
        _ragged_gemm[(routes.max_tiles * triton.cdiv(n, block_n),)](
            q, w_linear, scale_a, scale_b, out, routes.order, routes.offsets,
            routes.tile_experts, routes.tile_starts, n, k, routes.top_k,
            routes.block_m, block_n, group_size, source_is_token, 2,
            num_warps=4, num_stages=2,
        )
    return out


@triton.jit
def _pack_input_row_block(Q, SA, ORDER, COUNTS, OFFSETS, PACKED_Q, PACKED_SA,
                          E: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
                          C: tl.constexpr, GROUP: tl.constexpr,
                          BLOCK: tl.constexpr, ROWS: tl.constexpr):
    packed_rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    present = packed_rows < E * C
    experts = packed_rows // C
    local_rows = packed_rows % C
    counts = tl.load(COUNTS + experts, present, other=0)
    valid = present & (local_rows < counts)
    starts = tl.load(OFFSETS + experts, present, other=0)
    assignments = tl.load(ORDER + starts + local_rows, valid, other=0)
    tokens = assignments // R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Q + tokens[:, None] * K + cols[None, :],
                     valid[:, None] & (cols[None, :] < K), other=0)
    tl.store(PACKED_Q + packed_rows[:, None] * K + cols[None, :], values,
             present[:, None] & (cols[None, :] < K))
    groups = tl.program_id(1) * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP)
    scales = tl.load(SA + tokens[:, None] * (K // GROUP) + groups[None, :],
                     valid[:, None] & (groups[None, :] < K // GROUP), other=0)
    tl.store(PACKED_SA + packed_rows[:, None] * (K // GROUP) + groups[None, :], scales,
             present[:, None] & (groups[None, :] < K // GROUP))


def pack_inputs(q, scale_a, routes, capacity, group_size=256, *, rows_per_program=4):
    """Build zero-padded [E,C,K] inputs for an equal-capacity native batched GEMM.

    The caller guarantees capacity >= max(counts).  Routing and packing itself
    stay on the device; no padded row reads an original token.
    """
    k = q.shape[-1]
    packed_q = torch.empty((routes.experts, capacity, k), dtype=torch.int8,
                           device=q.device)
    packed_scale = torch.empty((routes.experts, capacity, k // group_size),
                               dtype=torch.float32, device=q.device)
    if capacity:
        _pack_input_row_block[(triton.cdiv(routes.experts * capacity, rows_per_program),
                               triton.cdiv(k, 1024))](
            q, scale_a, routes.order, routes.counts, routes.offsets,
            packed_q, packed_scale, routes.experts, k, routes.top_k,
            capacity, group_size, 1024, rows_per_program,
            num_warps=4,
        )
    return packed_q, packed_scale


@triton.jit
def _quantize_pack_input_row_block(X, INVERSE, KEYS, OFFSETS, PACKED_Q, PACKED_SA,
                                   T: tl.constexpr, E: tl.constexpr,
                                   K: tl.constexpr, R: tl.constexpr,
                                   C: tl.constexpr, GROUP: tl.constexpr,
                                   ROWS: tl.constexpr):
    tokens = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    present = tokens < T
    enabled = tl.full((ROWS,), False, tl.int1)
    destinations = ()
    route_masks = ()
    # A token is quantized once, including when several ranks select the same
    # expert. Each valid assignment has a unique inverse slot, so stores from
    # different tokens/ranks cannot collide.
    for rank in tl.static_range(R):
        assignment = tokens * R + rank
        slot = tl.load(INVERSE + assignment, present, other=-1)
        valid = present & (slot >= 0)
        expert = tl.load(KEYS + assignment, valid, other=0)
        valid = valid & (expert >= 0) & (expert < E)
        start = tl.load(OFFSETS + expert, valid, other=0)
        local_row = slot - start
        valid = valid & (local_row >= 0) & (local_row < C)
        destinations += (expert * C + local_row,)
        route_masks += (valid,)
        enabled = enabled | valid
    group = tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    # Mask before loading X: a masked token is allowed to contain NaNs.
    values = tl.load(X + tokens[:, None] * K + cols[None, :],
                     enabled[:, None], other=0).to(tl.float32)
    quantized, scales = _symmetric_quantize_row_block(values)
    for rank in tl.static_range(R):
        destination = destinations[rank]
        valid = route_masks[rank]
        tl.store(PACKED_Q + destination[:, None] * K + cols[None, :],
                 quantized, valid[:, None])
        tl.store(PACKED_SA + destination * (K // GROUP) + group, scales, valid)


@triton.jit
def _zero_packed_input_padding(COUNTS, PACKED_Q, PACKED_SA,
                               K: tl.constexpr, C: tl.constexpr,
                               GROUP: tl.constexpr, BLOCK: tl.constexpr,
                               ROWS: tl.constexpr):
    expert = tl.program_id(0)
    count = tl.load(COUNTS + expert)
    first = count + tl.program_id(1) * ROWS
    if first < C:
        local_rows = first + tl.arange(0, ROWS)
        packed_rows = expert * C + local_rows
        padding = local_rows < C
        cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        tl.store(PACKED_Q + packed_rows[:, None] * K + cols[None, :], 0,
                 padding[:, None] & (cols[None, :] < K))
        groups = tl.program_id(2) * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP)
        tl.store(PACKED_SA + packed_rows[:, None] * (K // GROUP) + groups[None, :],
                 0, padding[:, None] & (groups[None, :] < K // GROUP))


def quantize_pack_inputs(hidden_states, routes, capacity, group_size=256, *,
                         rows_per_program=8, padding_rows_per_program=16,
                         zero_initialize=False):
    """Native producer: quantize once and scatter to packed rows.

    Returns the exact layout and values of ``pack_inputs(*quantize_rows(x), ...)``
    without intermediate token INT8/scale tensors. Valid rows and padding are
    written by disjoint kernels; both buffers are fully initialized on every
    call, including CUDA graph replay with different routes. ``zero_initialize``
    permits benchmarking a full-buffer zero fill instead of padding-only stores.
    Thor native execution selects this producer for top-k up to eight, bounding
    the unrolled scatter's register usage. Other cases retain separate producers.
    """
    k = hidden_states.shape[-1]
    allocate = torch.zeros if zero_initialize else torch.empty
    packed_q = allocate((routes.experts, capacity, k), dtype=torch.int8,
                        device=hidden_states.device)
    packed_scale = allocate((routes.experts, capacity, k // group_size),
                            dtype=torch.float32, device=hidden_states.device)
    if capacity:
        if not zero_initialize:
            _zero_packed_input_padding[(routes.experts,
                                       triton.cdiv(capacity, padding_rows_per_program),
                                       triton.cdiv(k, 1024))](
                routes.counts, packed_q, packed_scale, k, capacity, group_size,
                1024, padding_rows_per_program, num_warps=4,
            )
        if routes.tokens:
            _quantize_pack_input_row_block[(triton.cdiv(routes.tokens, rows_per_program),
                                            k // group_size)](
                hidden_states, routes.inverse, routes.effective_ids, routes.offsets,
                packed_q, packed_scale, routes.tokens, routes.experts, k,
                routes.top_k, capacity, group_size, rows_per_program,
                num_warps=4, enable_fp_fusion=False,
            )
    return packed_q, packed_scale


@triton.jit
def _swiglu_quantize_row_block(GATE_UP, Q, S, OFFSETS, COUNTS,
                               M: tl.constexpr, I: tl.constexpr,
                               GROUP: tl.constexpr, ROWS: tl.constexpr,
                               HAS_ROUTES: tl.constexpr, E: tl.constexpr,
                               CAPACITY: tl.constexpr, INTERLEAVED: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    present = rows < M
    valid = present
    if HAS_ROUTES:
        if CAPACITY > 0:
            counts = tl.load(COUNTS + rows // CAPACITY, present, other=0)
            valid = present & (rows % CAPACITY < counts)
        else:
            valid = present & (rows < tl.load(OFFSETS + E))
    cols = group * GROUP + tl.arange(0, GROUP)
    # Native storage includes complete padding blocks. Avoid their nonlinear
    # math while still initializing every output row for the next GEMM's TMA.
    # Other layouts keep the original straight-line path at compile time.
    has_values = True
    if HAS_ROUTES and CAPACITY > 0:
        has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
    if has_values:
        gate_cols = 2 * cols if INTERLEAVED else cols
        up_cols = 2 * cols + 1 if INTERLEAVED else I + cols
        gate = tl.load(GATE_UP + rows[:, None] * (2 * I) + gate_cols[None, :],
                       valid[:, None], other=0).to(tl.float32)
        up = tl.load(GATE_UP + rows[:, None] * (2 * I) + up_cols[None, :],
                     valid[:, None], other=0).to(tl.float32)
        hidden = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
        quantized, scales = _symmetric_quantize_row_block(hidden)
    else:
        quantized = tl.full((ROWS, GROUP), 0, tl.int8)
        zero_scale = libdevice.div_rn(tl.full((), 1.0e-12, tl.float32), 127.0)
        scales = tl.broadcast_to(zero_scale, (ROWS,))
    tl.store(Q + rows[:, None] * I + cols[None, :], quantized, present[:, None])
    tl.store(S + rows * (I // GROUP) + group, scales, present)


def swiglu_quantize(gate_up, routes=None, *, packed_capacity=None, group_size=256,
                    rows_per_program=8, gate_up_layout="split"):
    """Compute SwiGLU in FP32 and directly quantize without a BF16 hidden round.

    Accept either [S,2I] ragged storage or [E,C,2I] native storage.  If routes is
    supplied, invalid/padded rows are read as zero rather than touching unwritten
    ragged GEMM output. The returned leading dimensions match gate_up. The
    final dimension is split [gate...,up...] or interleaved [g0,u0,g1,u1,...];
    the output hidden channels and their scale grouping retain the same order.
    """
    if gate_up_layout not in ("split", "interleaved"):
        raise ValueError("gate_up_layout must be 'split' or 'interleaved'")
    intermediate = gate_up.shape[-1] // 2
    rows = prod(gate_up.shape[:-1])
    q = torch.empty((*gate_up.shape[:-1], intermediate),
                    dtype=torch.int8, device=gate_up.device)
    scales = torch.empty((*gate_up.shape[:-1], intermediate // group_size),
                         dtype=torch.float32, device=gate_up.device)
    if rows:
        capacity = packed_capacity or (gate_up.shape[1] if gate_up.ndim == 3 else 0)
        _swiglu_quantize_row_block[(triton.cdiv(rows, rows_per_program), intermediate // group_size)](
            gate_up, q, scales, routes.offsets if routes else None,
            routes.counts if routes else None, rows, intermediate, group_size, rows_per_program,
            routes is not None, routes.experts if routes else 0, capacity,
            gate_up_layout == "interleaved",
            num_warps=4, enable_fp_fusion=False,
        )
    return q, scales


@triton.jit
def _combine(EXPERT_OUT, WEIGHTS, INVERSE, KEYS, OFFSETS, OUT,
              H: tl.constexpr, R: tl.constexpr, E: tl.constexpr,
              CAPACITY: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    token = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    assignment = token * R + ranks
    slot = tl.load(INVERSE + assignment, ranks < R, other=-1)
    valid = (ranks < R) & (slot >= 0)
    if CAPACITY > 0:
        expert = tl.load(KEYS + assignment, valid, other=0)
        valid = valid & (expert >= 0) & (expert < E)
        start = tl.load(OFFSETS + expert, valid, other=0)
        slot = expert * CAPACITY + slot - start
    values = tl.load(EXPERT_OUT + slot[:, None] * H + cols[None, :],
                     valid[:, None] & (cols[None, :] < H), other=0).to(tl.float32)
    weights = tl.load(WEIGHTS + assignment, valid, other=0).to(tl.float32)
    total = tl.sum(values * weights[:, None], axis=0)
    tl.store(OUT + token * H + cols, total, cols < H)


def combine(expert_output, topk_weights, routes, *, output_dtype,
            packed_capacity=None):
    """Apply the supplied routing weights and reduce in FP32 to [T,H]."""
    h = expert_output.shape[-1]
    out = torch.empty((routes.tokens, h), dtype=output_dtype, device=expert_output.device)
    if routes.tokens:
        capacity = packed_capacity or (expert_output.shape[1]
                                       if expert_output.ndim == 3 else 0)
        block_r = triton.next_power_of_2(routes.top_k)
        # Wide columns amortize the routing-index loads and launch far fewer
        # small CTAs. Bound the common tile to 8192 routed elements; retain the
        # existing narrow configuration for exceptionally large top-k values.
        block_h = (min(triton.next_power_of_2(min(h, 2048)), 8192 // block_r)
                   if block_r <= 64 else 128)
        _combine[(routes.tokens, triton.cdiv(h, block_h))](
            expert_output, topk_weights, routes.inverse, routes.effective_ids,
            routes.offsets, out, h, routes.top_k, routes.experts, capacity,
            block_r, block_h,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
