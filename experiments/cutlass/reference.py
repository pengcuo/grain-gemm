"""Unchanged sampled reference/check functions from archive/compare_optimized.py."""

def reference(a, b, sa, sb, torch):
    rows = torch.linspace(0, a.shape[0]-1, min(32, a.shape[0]), device=a.device).long()
    cols = torch.linspace(0, b.shape[1]-1, min(32, b.shape[1]), device=b.device).long()
    aa, bb = a[rows].cpu().int(), b[:, cols].cpu().int()
    saa, sbb = sa[rows].cpu(), sb[:, cols].cpu()
    expected = torch.zeros((len(rows), len(cols)), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], 256)):
        dot = aa[:, start:start+256] @ bb[start:start+256]
        scale = saa[:, group, None] * sbb[group, None, :]
        expected = (dot.double()*scale.double()+expected.double()).float()
    return rows, cols, expected.to(torch.bfloat16)


def check(out, ref, torch):
    rows, cols, expected = ref
    actual = out[rows][:, cols].cpu()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(out).all().item()
    return dict(checked_rows=len(rows), checked_columns=len(cols),
                max_abs_error=(actual.float()-expected.float()).abs().max().item())
