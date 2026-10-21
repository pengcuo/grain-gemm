"""Packed-input producer contracts against independent CPU quantization."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from grain_gemm.kernels import moe as kernels, moe_input_io
from test_moe_ffn import cuda, quantize_rows_cpu


def _reference(x, ids, mask, experts, capacity):
    """Enumerate original token/rank order without GPU sorting or packing."""
    assert x.device.type == ids.device.type == mask.device.type == "cpu"
    k = x.shape[-1]
    q = torch.zeros((experts, capacity, k), dtype=torch.int8)
    scales = torch.zeros((experts, capacity, k // 256), dtype=torch.float32)
    counts = [0] * experts
    for token in range(x.shape[0]):
        if not mask[token]:
            continue
        values, scale = quantize_rows_cpu(x[token:token + 1])
        for expert in ids[token].tolist():
            if not 0 <= expert < experts:
                continue
            row = counts[expert]
            assert row < capacity
            q[expert, row], scales[expert, row] = values[0], scale[0]
            counts[expert] += 1
    return q, scales


def _assert_bits(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.is_contiguous()
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        assert torch.equal(actual_tensor.cpu().view(torch.uint8),
                           expected_tensor.view(torch.uint8))


def _offset_cuda(value, elements=16):
    backing = torch.empty(value.numel() + elements, device="cuda", dtype=value.dtype)
    view = backing[elements:].view(value.shape)
    view.copy_(value)
    assert view.storage_offset() == elements
    return view


@cuda
@pytest.mark.parametrize("dtype,zero_initialize", [(torch.bfloat16, False),
                                                  (torch.float16, True)])
def test_duplicate_routes_masked_nan_and_tail_groups(dtype, zero_initialize):
    """Odd group counts, partial capacity and duplicate ranks retain all bits."""
    tokens, experts, capacity, k = 19, 5, 17, 768
    generator = torch.Generator().manual_seed(20261020)
    x = torch.randn((tokens, k), generator=generator).to(dtype)
    x[:, ::256] = 127
    ties = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=dtype)
    x[:, 1::256] = ties[torch.arange(tokens) % ties.numel(), None]
    x[1].zero_()
    choices = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 2, 0]], dtype=torch.int64)
    ids = choices[torch.arange(tokens) % 3].contiguous()
    mask = torch.ones(tokens, dtype=torch.bool)
    mask[[0, 7, 18]] = False
    x[~mask] = float("nan")
    ids[~mask] = torch.tensor([2**40, -(2**40), 2**63 - 1])
    ids[3, 1] = -(2**63)  # An invalid unmasked rank is ignored by the producer.
    expected = _reference(x, ids, mask, experts, capacity)
    gx, gids, gmask = (_offset_cuda(value) for value in (x, ids, mask))
    routes = kernels.prepare_routes(gids, experts, gmask, include_tiles=False)
    routes = replace(routes, offsets=_offset_cuda(routes.offsets, 1))
    actual = moe_input_io.quantize_pack_inputs_scatter(
        gx, routes, capacity, zero_initialize=zero_initialize,
    )
    _assert_bits(actual, expected)


@cuda
@pytest.mark.parametrize("rows,groups", [(4, 2), (1, 8)])
def test_four_byte_output_scale_offset(rows, groups):
    """Adjacent scale stores require scalar alignment, including the tail tile."""
    x = torch.zeros((3, 768), dtype=torch.bfloat16)
    x[0, ::256] = 127
    x[0, 1::256] = 2.5
    ids = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int64)
    mask = torch.ones(3, dtype=torch.bool)
    experts, capacity = 2, 9
    expected = _reference(x, ids, mask, experts, capacity)
    gx = _offset_cuda(x)
    routes = kernels.prepare_routes(ids.cuda(), experts, include_tiles=False)
    routes = replace(routes, offsets=_offset_cuda(routes.offsets, 1))
    q = _offset_cuda(torch.full_like(expected[0], -128))
    scales = _offset_cuda(torch.full_like(expected[1], float("nan")), 1)
    assert scales.data_ptr() % 16 == 4
    moe_input_io._scatter_quantize_and_pad[
        (triton.cdiv(routes.tokens, rows) + experts, triton.cdiv(768 // 256, groups))
    ](gx, routes.inverse, routes.effective_ids, routes.offsets, routes.counts,
      q, scales, routes.tokens, experts, 768, routes.top_k, capacity, 256, False,
      rows, groups,
      num_warps=4, enable_fp_fusion=False)
    _assert_bits((q, scales), expected)


@cuda
@pytest.mark.parametrize("kind", ["empty", "all_masked"])
def test_no_valid_tokens_still_initializes_padding(kind):
    tokens = 0 if kind == "empty" else 9
    x = torch.full((tokens, 768), float("nan"), dtype=torch.bfloat16)
    ids = torch.full((tokens, 3), 2**63 - 1, dtype=torch.int64)
    mask = torch.zeros(tokens, dtype=torch.bool)
    routes = kernels.prepare_routes(ids.cuda(), 5, mask.cuda(), include_tiles=False)
    actual = moe_input_io.quantize_pack_inputs_scatter(x.cuda(), routes, 11)
    _assert_bits(actual, _reference(x, ids, mask, 5, 11))


@cuda
def test_graph_replay_replaces_routes_and_clears_previous_rows():
    """Moving every route to an empty expert must also clear old live rows."""
    generator = torch.Generator().manual_seed(20261020)
    x = torch.randn((11, 768), generator=generator).bfloat16().cuda()
    ids = (torch.arange(33).reshape(11, 3) % 3).cuda()
    mask = torch.ones(11, dtype=torch.bool, device="cuda")
    experts, capacity = 5, 33

    def produce():
        routes = kernels.prepare_routes(ids, experts, mask, include_tiles=False)
        return moe_input_io.quantize_pack_inputs_scatter(x, routes, capacity)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            produce()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = produce()
    graph.replay()
    _assert_bits(captured, _reference(x.cpu(), ids.cpu(), mask.cpu(), experts, capacity))
    ids.fill_(4)
    graph.replay()
    _assert_bits(captured, _reference(x.cpu(), ids.cpu(), mask.cpu(), experts, capacity))
    mask.zero_()
    x.fill_(float("nan"))
    graph.replay()
    _assert_bits(captured, _reference(x.cpu(), ids.cpu(), mask.cpu(), experts, capacity))
    mask.fill_(True)
    x.fill_(1)
    ids.fill_(0)
    graph.replay()
    _assert_bits(captured, _reference(x.cpu(), ids.cpu(), mask.cpu(), experts, capacity))
