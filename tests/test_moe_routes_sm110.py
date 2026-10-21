"""Thor route selection and independent stable metadata references."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm.kernels import moe as kernels


@pytest.mark.parametrize("capability,slots,experts,expected", [
    ((11, 0), 1, 32, 2048),
    ((11, 0), 65537, 32, 2048),
    ((11, 0), 262144, 64, 2048),
    ((12, 0), 65536, 64, 512),
    ((12, 1), 65537, 64, 0),
    ((8, 0), 65536, 32, 512),
    ((8, 0), 65537, 32, 0),
    ((11, 1), 65536, 32, 512),
])
def test_routing_chunks_are_selected_for_the_tensor_device(
        monkeypatch, capability, slots, experts, expected):
    """Architecture gating retains GB10/other-device routing thresholds."""
    device = torch.device("cuda:3")
    calls = []

    def get_capability(selected):
        calls.append(selected)
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    assert kernels._routing_chunk_size(slots, experts, device) == expected
    assert calls == [device]


@pytest.mark.parametrize("slots,experts", [(0, 32), (262145, 64), (65536, 65)])
def test_large_or_empty_routes_keep_the_general_sort_without_device_query(
        monkeypatch, slots, experts):
    def unexpected_capability(*args):
        pytest.fail("The general-sort cases need no architecture query")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability)
    assert kernels._routing_chunk_size(slots, experts, torch.device("cuda:3")) == 0


@pytest.mark.parametrize("experts,tokens,include_tiles", [
    (32, 32769, False),  # Above the old 65536-assignment cutoff, partial chunk.
    (64, 65536, True),   # Exact new cutoff and largest supported expert count.
])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_large_thor_routes_match_stable_cpu_metadata(experts, tokens, include_tiles, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("The extended counting-sort path requires SM110")

    top_k = 4
    ids = ((torch.arange(tokens * top_k, dtype=torch.int64) * 17 + 11)
           % experts).reshape(tokens, top_k).to(dtype)
    ids[:, 1] = ids[:, 0]  # Duplicate expert slots remain separate assignments.
    mask = torch.arange(tokens) % 11 != 0
    limits = torch.iinfo(dtype)
    flat = ids.view(-1)
    # Include chunk boundaries, signed extremes and both masked/unmasked IDs.
    for slot, value in ((1, limits.max), (2047, limits.min), (2048, experts),
                        (65535, -1), (65536, limits.max)):
        flat[slot] = value
    mask[0] = True
    mask[2048 // top_k] = False

    enabled = mask[:, None].expand_as(ids).reshape(-1)
    in_range = (flat >= 0) & (flat < experts)
    valid = enabled & in_range
    keys = torch.where(valid, flat, experts)
    order = torch.argsort(keys, stable=True)
    counts = torch.bincount(keys[valid].long(), minlength=experts).int()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
    total = int(counts.sum())
    inverse = torch.full((flat.numel(),), -1, dtype=torch.int32)
    inverse[order[:total]] = torch.arange(total, dtype=torch.int32)
    statistics = torch.tensor([int((enabled & ~in_range).sum()), int(counts.max()), total],
                              dtype=torch.int32)

    routes = kernels.prepare_routes(ids.cuda(), experts, mask.cuda(),
                                    include_tiles=include_tiles)
    for actual, expected in ((routes.effective_ids, keys), (routes.order, order),
                              (routes.inverse, inverse), (routes.counts, counts),
                              (routes.offsets, offsets), (routes.statistics, statistics)):
        assert torch.equal(actual.cpu(), expected)
    if not include_tiles:
        assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
        return

    tiles = (counts + routes.block_m - 1) // routes.block_m
    tile_offsets = torch.cat((torch.zeros(1, dtype=torch.int32), tiles.cumsum(0).int()))
    assert torch.equal(routes.tile_offsets.cpu(), tile_offsets)
    expected_experts, expected_starts = [], []
    for expert, count in enumerate(tiles.tolist()):
        expected_experts.extend([expert] * count)
        expected_starts.extend(int(offsets[expert]) + tile * routes.block_m
                               for tile in range(count))
    active = len(expected_experts)
    assert routes.tile_experts[:active].cpu().tolist() == expected_experts
    assert routes.tile_starts[:active].cpu().tolist() == expected_starts
    assert (routes.tile_experts[active:] < 0).all().item()
