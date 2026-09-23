"""Unchanged full-output fixture from archive/sanitize_cutlass.py."""

import torch


def case(m, n, k):
    groups = k // 256
    rows, cols = torch.arange(m)[:, None], torch.arange(n)[:, None]
    group = torch.arange(groups)[None, :]
    endpoints = torch.tensor([-128, 127], dtype=torch.int8)
    a_group = endpoints[(rows + group) % 2]
    b_group = endpoints[(cols + group // 2) % 2]
    # Prime periods expose missing row/column CTA offsets; every group also
    # carries different positive, negative, and zero scales.
    sa = ((3 * rows + 5 * group) % 23 - 11).float() * 0.125
    sb = (((7 * cols + 11 * group) % 29 - 14).float() * 0.0625).t().contiguous()
    expected = torch.zeros((m, n), dtype=torch.float32)
    for g in range(groups):
        # Each fixture group is constant along K, so its exact INT32 dot is
        # independently available without a large CPU GEMM.
        partial = a_group[:, g, None].int() * b_group[None, :, g].int() * 256
        scales = sa[:, g, None] * sb[g, None, :]
        expected = (expected.double() + partial.double() * scales.double()).float()
    a = a_group.repeat_interleave(256, dim=1).cuda()
    b = b_group.repeat_interleave(256, dim=1).cuda().t()
    return a, b, sa.cuda(), sb.cuda(), expected.to(device="cuda", dtype=torch.bfloat16)
