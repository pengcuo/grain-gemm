# SPDX-License-Identifier: Apache-2.0
"""Packed G256 inputs with token quantization and padding in one launch.

Each token is quantized once and scattered to all its expert assignments.
Processing adjacent K groups together amortizes routing loads and writes adjacent
scale values. The caller selects this producer where measured performance
justifies it; architecture and size dispatch remain in the FFN implementation.
"""

import torch
import triton
import triton.language as tl

from .moe import _symmetric_quantize_row_block


@triton.jit
def _quantize_groups(values, GROUP: tl.constexpr, ROWS: tl.constexpr,
                     GROUPS: tl.constexpr):
    # Keep each reduction over the original 256 adjacent K elements and reuse
    # the existing RN division and nearest-even quantization implementation.
    flat = tl.reshape(values, (ROWS * GROUPS, GROUP))
    quantized, scales = _symmetric_quantize_row_block(flat)
    return (tl.reshape(quantized, (ROWS, GROUPS, GROUP)),
            tl.reshape(scales, (ROWS, GROUPS)))


@triton.jit
def _scatter_quantize_and_pad(X, INVERSE, KEYS, OFFSETS, COUNTS, Q, SA,
                              T: tl.constexpr, E: tl.constexpr,
                              K: tl.constexpr, R: tl.constexpr,
                              C: tl.constexpr, GROUP: tl.constexpr,
                              ZERO_INITIALIZED: tl.constexpr,
                              ROWS: tl.constexpr = 4, GROUPS: tl.constexpr = 2,
                              SCALE_X=None, PREQUANTIZED: tl.constexpr = False):
    program = tl.program_id(0)
    token_programs = tl.cdiv(T, ROWS)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    group_valid = groups < K // GROUP
    cols = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]

    if program < token_programs:
        tokens = program * ROWS + tl.arange(0, ROWS)
        present = tokens < T
        enabled = tl.full((ROWS,), False, tl.int1)
        destinations = ()
        masks = ()
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
            masks += (valid,)
            enabled = enabled | valid
        values = tl.load(
            X + tokens[:, None, None] * K + cols[None, :, :],
            enabled[:, None, None] & group_valid[None, :, None], other=0,
        )
        if PREQUANTIZED:
            quantized = values
            scales = tl.load(
                SCALE_X + tokens[:, None] * (K // GROUP) + groups[None, :],
                enabled[:, None] & group_valid[None, :], other=0,
            )
        else:
            quantized, scales = _quantize_groups(values.to(tl.float32), GROUP, ROWS, GROUPS)
        for rank in tl.static_range(R):
            destination = destinations[rank]
            valid = masks[rank]
            tl.store(
                Q + destination[:, None, None] * K + cols[None, :, :],
                quantized, valid[:, None, None] & group_valid[None, :, None],
            )
            tl.store(
                SA + destination[:, None] * (K // GROUP) + groups[None, :],
                scales, valid[:, None] & group_valid[None, :],
            )
    elif not ZERO_INITIALIZED:
        # One worker per expert/group tile walks only its padding tail. Live
        # assignments and padding are disjoint, requiring no inter-CTA barrier.
        padding_expert = program - token_programs
        first = tl.load(COUNTS + padding_expert)
        rows = tl.arange(0, 64)
        while first < C:
            local_rows = first + rows
            packed_rows = padding_expert * C + local_rows
            padding_valid = local_rows < C
            tl.store(
                Q + packed_rows[:, None, None] * K + cols[None, :, :],
                0, padding_valid[:, None, None] & group_valid[None, :, None],
            )
            tl.store(
                SA + packed_rows[:, None] * (K // GROUP) + groups[None, :],
                0.0, padding_valid[:, None] & group_valid[None, :],
            )
            first += 64


def quantize_pack_inputs_scatter(hidden_states, routes, capacity, group_size=256, *,
                                 zero_initialize=False):
    """Return fully initialized INT8 ``[E,C,K]`` and FP32 ``[E,C,K/256]``.

    Values, positive-zero padding, stable route order and contiguous strides
    match ``moe.quantize_pack_inputs``. Masked tokens are not loaded, including
    when their activations contain NaNs. ``zero_initialize`` retains the option
    to fill both buffers first instead of fusing padding into the scatter.
    """
    k = hidden_states.shape[-1]
    if group_size != 256 or k % group_size:
        raise ValueError("The input producer requires complete G256 groups")
    allocate = torch.zeros if zero_initialize else torch.empty
    q = allocate((routes.experts, capacity, k), dtype=torch.int8,
                 device=hidden_states.device)
    scales = allocate((routes.experts, capacity, k // group_size),
                      dtype=torch.float32, device=hidden_states.device)
    if capacity:
        # On the measured large Thor shape, keeping all six groups of a token
        # together reduces repeated route loads. Eight lanes include two masked
        # groups; the numerical reduction remains over exactly G256 elements.
        rows, groups = ((1, 8) if (routes.tokens, routes.experts, routes.top_k, k)
                        == (32768, 32, 4, 1536) else (4, 2))
        programs = triton.cdiv(routes.tokens, rows)
        if not zero_initialize:
            programs += routes.experts
        if programs:
            _scatter_quantize_and_pad[(programs, triton.cdiv(k // group_size, groups))](
                hidden_states, routes.inverse, routes.effective_ids, routes.offsets,
                routes.counts, q, scales, routes.tokens, routes.experts, k,
                routes.top_k, capacity, group_size, zero_initialize,
                rows, groups,
                num_warps=4, enable_fp_fusion=False,
            )
    return q, scales


def pack_inputs_scatter(hidden_states, scale_x, routes, capacity, group_size=256):
    """Scatter prequantized token rows/scales to experts without recomputing either.

    Duplicate routes keep separate rows, masked inputs/scales are not loaded,
    and unused expert rows receive positive-zero data and scales.
    """
    k = hidden_states.shape[-1]
    if group_size != 256 or k % group_size:
        raise ValueError("The input producer requires complete G256 groups")
    q = torch.empty((routes.experts, capacity, k), dtype=torch.int8,
                    device=hidden_states.device)
    scales = torch.empty((routes.experts, capacity, k // group_size),
                         dtype=torch.float32, device=hidden_states.device)
    if capacity:
        rows, groups = 4, triton.next_power_of_2(min(k // group_size, 8))
        _scatter_quantize_and_pad[(triton.cdiv(routes.tokens, rows) + routes.experts,
                                  triton.cdiv(k // group_size, groups))](
            hidden_states, routes.inverse, routes.effective_ids, routes.offsets,
            routes.counts, q, scales, routes.tokens, routes.experts, k,
            routes.top_k, capacity, group_size, False, rows, groups,
            scale_x, True, num_warps=4, enable_fp_fusion=False,
        )
    return q, scales
