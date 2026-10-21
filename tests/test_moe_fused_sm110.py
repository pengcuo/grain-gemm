"""Fused Thor G1 numerical, stream, replay and public dispatch regressions.

GPU tests deliberately use the existing native G1 plus activation quantizer
as the bit reference. Public cases use transposed Linear weight views and
exercise the same full FFN with and without the optional fused library.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import moe_ffn
from grain_gemm.kernels import cutlass_moe, moe as kernels, moe_fused
from grain_gemm.moe import _use_fused_gate_up


@pytest.fixture(scope="module")
def thor_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("The fused MoE library requires Thor SM110")
    if not cutlass_moe.is_available() or not moe_fused.is_available():
        pytest.skip("Build both optional Thor MoE libraries")


@pytest.mark.parametrize("tokens", [4096, 8192, 16384])
def test_selector_accepts_measured_shapes_and_selected_device(monkeypatch, tokens):
    device = object()
    observed = []

    def available(selected):
        observed.append(selected)
        return True

    monkeypatch.setattr(moe_fused, "is_available", available)
    assert _use_fused_gate_up(tokens, 1536, 1024, 32, 4, device, capturing=False)
    assert observed == [device]


@pytest.mark.parametrize("shape,capturing", [
    ((16384, 1536, 1024, 32, 4), True),
    ((32768, 1536, 1024, 32, 4), False),
    ((4096, 1280, 1024, 32, 4), False),
    ((4096, 1536, 768, 32, 4), False),
    ((4096, 1536, 1024, 16, 4), False),
    ((4096, 1536, 1024, 32, 2), False),
    ((4096, 1536, 1024, 32, 4), True),
])
def test_selector_retains_unmeasured_and_graph_paths(monkeypatch, shape, capturing):
    monkeypatch.setattr(moe_fused, "is_available",
                        lambda device: pytest.fail("Unselected shapes must not load the optional library"))
    assert not _use_fused_gate_up(*shape, object(), capturing=capturing)


def test_selector_retains_absent_optional_build(monkeypatch):
    monkeypatch.setattr(moe_fused, "is_available", lambda device: False)
    assert not _use_fused_gate_up(4096, 1536, 1024, 32, 4, object(), capturing=False)


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def _packed_case(seed=20261021):
    """Cross both 64-row task boundaries and G256 activation boundaries."""
    e, c, h, i = 7, 128, 512, 512
    generator = torch.Generator().manual_seed(seed)
    a = torch.randint(-127, 128, (e, c, h), dtype=torch.int8, generator=generator).cuda()
    w = torch.randint(-127, 128, (e, 2 * i, h), dtype=torch.int8, generator=generator).cuda()
    w[:, :, 0] = -128
    w[:, :, 1] = 127
    sa = (torch.rand((e, c, h // 256), generator=generator) * .008 + .001).cuda()
    sb = (torch.rand((e, h // 256, 2 * i), generator=generator) * .006 + .001).cuda()
    sb[:, :, ::29] = 0
    # Include a valid all-zero input row as well as padding.
    a[:, 0].zero_()
    count_storage = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    counts = count_storage[1:]
    counts.copy_(torch.tensor([0, 1, 63, 64, 65, 127, 128], dtype=torch.int32))
    q = torch.full((e, c, i), -128, dtype=torch.int8, device="cuda")
    s = torch.full((e, c, i // 256), float("nan"), dtype=torch.float32, device="cuda")
    return a, w, sa, sb, counts, q, s


def _reference_packed(a, weight, scale_a, scale_b, counts):
    e, c, _ = a.shape
    gate_up = torch.empty((e, c, weight.shape[1]), dtype=torch.bfloat16, device=a.device)
    cutlass_moe._launch_prevalidated(a, weight, scale_a, scale_b, gate_up,
                                    config_id=34, counts=counts)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=a.device),
                         counts.cumsum(0, dtype=torch.int32)))
    routes = SimpleNamespace(counts=counts, offsets=offsets, experts=e)
    return kernels.swiglu_quantize(gate_up, routes, packed_capacity=c)


def _check_packed_result(case):
    a, weight, scale_a, scale_b, counts, q, s = case
    expected_q, expected_s = _reference_packed(a, weight, scale_a, scale_b, counts)
    _assert_bits(q, expected_q)
    _assert_bits(s, expected_s)
    padding = torch.arange(a.shape[1], device=a.device)[None, :] >= counts[:, None]
    assert torch.count_nonzero(q[padding]).item() == 0
    tiny_scale = (torch.tensor(1.e-12, dtype=torch.float32) / 127).cuda()
    assert torch.equal(s[padding].contiguous().view(torch.int32),
                       tiny_scale.expand_as(s[padding]).contiguous().view(torch.int32))


@pytest.mark.parametrize("consumer_warps", [8, 16])
def test_gpu_fused_gate_up_matches_all_reference_bits_including_padding(thor_native, consumer_warps):
    case = _packed_case()
    moe_fused._launch_prevalidated(*case, consumer_warps=consumer_warps, persistent_blocks=3)
    _check_packed_result(case)


def test_gpu_fused_graph_replay_replaces_counts_and_clears_old_rows(thor_native):
    case = _packed_case()
    counts, q, s = case[4:]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            moe_fused._launch_prevalidated(*case, persistent_blocks=3)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        moe_fused._launch_prevalidated(*case, persistent_blocks=3)
    for population in ([0, 1, 63, 64, 65, 127, 128],
                       [128, 65, 64, 63, 1, 0, 127], [0] * 7):
        counts.copy_(torch.tensor(population, dtype=torch.int32))
        q.fill_(-128)
        s.fill_(float("nan"))
        graph.replay()
        _check_packed_result(case)


def test_gpu_fused_calls_on_two_nondefault_streams_have_private_outputs(thor_native):
    cases = [_packed_case(seed) for seed in (20261021, 20261022)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    current = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(current)
    for case, stream in zip(cases, streams):
        with torch.cuda.stream(stream):
            moe_fused._launch_prevalidated(*case, persistent_blocks=3)
    for stream in streams:
        current.wait_stream(stream)
    for case in cases:
        _check_packed_result(case)


@pytest.fixture(scope="module")
def public_inputs(thor_native):
    """Shared immutable weights retain the public transposed Linear layout."""
    generator = torch.Generator().manual_seed(20261023)
    experts, h, i, tokens = 32, 1536, 1024, 16384
    x = torch.randn((tokens, h), generator=generator).bfloat16().cuda()
    weights = []
    scales = []
    for n, k in ((2 * i, h), (h, i)):
        linear = torch.randint(-32, 33, (experts, n, k), dtype=torch.int8,
                               generator=generator).cuda()
        linear[:, :, 0] = -128
        weights.append(linear.transpose(-2, -1))
        scale = (torch.rand((experts, k // 256, n), generator=generator) * .002 + .001).cuda()
        scale[:, :, ::73] = 0
        scales.append(scale)
    ids = ((torch.arange(tokens)[:, None] * 5 + torch.tensor([0, 7, 13, 19])) % experts).cuda()
    scores = torch.tensor([.5, -.25, 1.5, 0.], dtype=torch.float32).repeat(tokens, 1).cuda()
    return dict(hidden_states=x, w_gate_up=weights[0], w_down=weights[1],
                topk_weights=scores, topk_ids=ids,
                scale_gate_up=scales[0], scale_down=scales[1])


def _public_case(inputs, tokens):
    return {name: tensor[:tokens] if name in ("hidden_states", "topk_ids", "topk_weights")
            else tensor for name, tensor in inputs.items()}


def _without_fused(case, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(moe_fused, "is_available", lambda device=None: False)
        return moe_ffn(**case, backend="auto")


@pytest.mark.parametrize("tokens", [4096, 8192, 16384])
def test_gpu_public_auto_selects_fused_and_matches_full_ffn_bits(public_inputs, monkeypatch, tokens):
    case = _public_case(public_inputs, tokens)
    expected = _without_fused(case, monkeypatch)
    launch = moe_fused._launch_prevalidated
    calls = []

    def observed(*args, **kwargs):
        calls.append((args[0].device, torch.cuda.current_stream(args[0].device).cuda_stream))
        return launch(*args, **kwargs)

    monkeypatch.setattr(moe_fused, "_launch_prevalidated", observed)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = moe_ffn(**case, backend="auto")
    torch.cuda.current_stream().wait_stream(stream)
    assert calls == [(case["hidden_states"].device, stream.cuda_stream)]
    _assert_bits(actual, expected)


def test_gpu_public_fused_preserves_duplicate_routes_masked_nan_and_invalid_ids(public_inputs, monkeypatch):
    case = _public_case(public_inputs, 4096)
    case["hidden_states"] = case["hidden_states"].clone()
    case["topk_ids"] = case["topk_ids"].clone()
    case["topk_weights"] = case["topk_weights"].clone()
    case["topk_ids"][:, 1] = case["topk_ids"][:, 0]
    mask = torch.arange(4096, device="cuda") % 11 != 0
    case["token_mask"] = mask
    case["hidden_states"][~mask] = float("nan")
    case["topk_weights"][~mask] = float("nan")
    case["topk_ids"][~mask] = torch.tensor([2**63 - 1, -(2**63), 2**40, -(2**40)], device="cuda")
    expected = _without_fused(case, monkeypatch)
    launch = moe_fused._launch_prevalidated
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(moe_fused, "_launch_prevalidated", observed)
    actual = moe_ffn(**case, backend="auto")
    assert calls == [True]
    _assert_bits(actual, expected)
    assert torch.count_nonzero(actual[~mask]).item() == 0


def _unexpected_fused(*args, **kwargs):
    pytest.fail("This public execution must retain its fallback path")


def test_gpu_public_optional_absence_retains_native_ffn(public_inputs, monkeypatch):
    case = _public_case(public_inputs, 4096)
    expected = _without_fused(case, monkeypatch)
    monkeypatch.setattr(moe_fused, "is_available", lambda device=None: False)
    monkeypatch.setattr(moe_fused, "_launch_prevalidated", _unexpected_fused)
    actual = moe_ffn(**case, backend="auto")
    _assert_bits(actual, expected)


def test_gpu_public_fp16_auto_retains_triton_fallback(public_inputs, monkeypatch):
    case = _public_case(public_inputs, 4096)
    case["hidden_states"] = case["hidden_states"].half()
    monkeypatch.setattr(moe_fused, "_launch_prevalidated", _unexpected_fused)
    expected = moe_ffn(**case, backend="triton")
    actual = moe_ffn(**case, backend="auto")
    _assert_bits(actual, expected)


def test_gpu_public_auto_graph_replay_retains_triton_and_dynamic_routing(public_inputs, monkeypatch):
    case = _public_case(public_inputs, 4096)
    for name in ("hidden_states", "topk_ids", "topk_weights"):
        case[name] = case[name].clone()
    case["token_mask"] = torch.ones(4096, dtype=torch.bool, device="cuda")
    monkeypatch.setattr(moe_fused, "_launch_prevalidated", _unexpected_fused)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            moe_ffn(**case, backend="triton")
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = moe_ffn(**case, backend="auto")
    graph.replay()
    _assert_bits(captured, moe_ffn(**case, backend="triton"))
    # A single previously idle expert now receives every top-k slot. Captured
    # auto must use the ragged path rather than any warmup expert capacity.
    case["topk_ids"].fill_(31)
    case["topk_weights"].fill_(.25)
    case["token_mask"][::7] = False
    case["hidden_states"][~case["token_mask"]] = float("nan")
    graph.replay()
    _assert_bits(captured, moe_ffn(**case, backend="triton"))
    case["token_mask"].zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0


def test_gpu_public_unmasked_invalid_ids_still_raise_before_fused(public_inputs, monkeypatch):
    case = _public_case(public_inputs, 4096)
    case["topk_ids"] = case["topk_ids"].clone()
    case["topk_ids"][0, 0] = 2**63 - 1
    monkeypatch.setattr(moe_fused, "_launch_prevalidated", _unexpected_fused)
    with pytest.raises(ValueError, match="unmasked topk_ids"):
        moe_ffn(**case, backend="auto")
