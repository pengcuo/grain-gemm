"""MoE FFN contracts and independent, per-token CPU INT32 references.

The reference deliberately does not sort, pad, or pack tokens by expert. Each
top-k slot is evaluated separately, including repeated expert IDs.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import moe_ffn


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="An SM80-or-newer CUDA device is required",
)


def quantize_rows_cpu(x, group_size=256):
    """Independent symmetric row/group quantizer, including ties-to-even."""
    assert x.device.type == "cpu"
    blocks = x.float().reshape(x.shape[0], -1, group_size)
    scales = blocks.abs().amax(-1).clamp_min(1e-12) / 127
    values = torch.round(blocks / scales.unsqueeze(-1)).clamp(-127, 127)
    return values.to(torch.int8).reshape(x.shape), scales


def group_dot_cpu(a, b, scale_a, scale_b):
    """INT32 dots, FP32 scale products and FP32 group accumulation.

    FP64 multiply-add followed by FP32 conversion approximates the GEMM FMA
    without using a floating-point GEMM as the integer-dot reference.
    """
    assert a.device.type == b.device.type == "cpu"
    result = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], 256)):
        partial = a[:, start : start + 256].int() @ b[start : start + 256].int()
        scales = scale_a[:, group, None] * scale_b[group, None, :]
        result = (result.double() + partial.double() * scales.double()).float()
    return result


def reference(case):
    """Evaluate every route independently, retaining the public rounding points."""
    host = {name: value.cpu() for name, value in case.items()}
    x, ids, scores = (host[name] for name in ("hidden_states", "topk_ids", "topk_weights"))
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        if "token_mask" in host and not host["token_mask"][token]:
            continue
        qa, sa = quantize_rows_cpu(x[token : token + 1])
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            first = group_dot_cpu(
                qa, host["w_gate_up"][expert], sa, host["scale_gate_up"][expert]
            ).to(x.dtype).float()
            gate, up = first.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            qhidden, shidden = quantize_rows_cpu(activated)
            second = group_dot_cpu(
                qhidden, host["w_down"][expert], shidden, host["scale_down"][expert]
            ).to(x.dtype).float()
            output[token] += second[0] * scores[token, slot]
    return output.to(x.dtype)


def make_case(*, dtype=torch.bfloat16, tokens=7, h=512, i=256, experts=5, device="cuda"):
    """Vary experts, rows, channels and both reduction groups independently."""
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((tokens, h), generator=generator).to(dtype)
    weights = []
    scales = []
    for out_features, in_features in ((2 * i, h), (h, i)):
        original = torch.randn(
            (experts * out_features, in_features), generator=generator
        ) / in_features**0.5
        q, s = quantize_rows_cpu(original)
        q = q.reshape(experts, out_features, in_features)
        s = s.reshape(experts, out_features, in_features // 256)
        # Weight INT8 accepts -128, although activation quantization uses ±127.
        q[:, :, 0] = -128
        q[:, :, 1] = 127
        s[:, ::11, :] = 0
        weights.append(q.transpose(-2, -1))
        scales.append(s.transpose(-2, -1).contiguous())
    routes = torch.tensor(
        [[0, 1, 0], [2, 0, 1], [1, 2, 2], [0, 2, 1], [2, 2, 0], [1, 0, 2], [0, 1, 2]],
        dtype=torch.int64,
    )
    ids = routes[torch.arange(tokens) % routes.shape[0]].contiguous()
    # Deliberately not normalized; one slot contributes zero.
    scores = torch.tensor([0.25, 1.5, 0.0]).repeat(tokens, 1)
    case = dict(
        hidden_states=x,
        w_gate_up=weights[0],
        w_down=weights[1],
        topk_weights=scores,
        topk_ids=ids,
        scale_gate_up=scales[0],
        scale_down=scales[1],
    )
    # .to preserves the transposed weight layouts.
    return {name: value.to(device) for name, value in case.items()}


def assert_reference(actual, case):
    expected = reference(case)
    assert actual.shape == case["hidden_states"].shape
    assert actual.dtype == case["hidden_states"].dtype
    assert actual.stride() == (actual.shape[1], 1)
    # Independent FP32 activation implementations can differ around an INT8
    # rounding boundary; the absolute bound also covers outputs near zero.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.025)


def require_backend(backend, dtype):
    if backend not in ("cuda", "cutlass"):
        return
    from grain_gemm.kernels import cuda as native_cuda, cutlass

    if dtype != torch.bfloat16 or torch.cuda.get_device_capability() not in ((11, 0), (12, 0), (12, 1)):
        pytest.skip("Native INT8 MoE requires a supported device and BF16 output")
    native = cutlass if backend == "cutlass" else native_cuda
    if not native.is_available():
        pytest.skip(f"Build the optional {backend} backend")


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton", "cuda", "cutlass"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ragged_routes_empty_experts_extrema_and_nonunit_scores(backend, dtype):
    require_backend(backend, dtype)
    case = make_case(dtype=dtype)
    case["token_mask"] = torch.tensor(
        [True, False, True, True, False, True, True], device="cuda"
    )
    saved = {name: value.clone() for name, value in case.items()}
    actual = moe_ffn(**case, backend=backend)
    assert_reference(actual, case)
    assert torch.count_nonzero(actual[~case["token_mask"]]).item() == 0
    for name, value in case.items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


@cuda
def test_thor_auto_selects_native_for_balanced_routing(monkeypatch):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Thor automatic native dispatch requires SM110")
    require_backend("cutlass", torch.bfloat16)
    import grain_gemm.moe as implementation

    case = make_case(tokens=256, h=256, experts=2)
    case["topk_ids"] = (torch.arange(256, device="cuda") % 2).reshape(256, 1)
    case["topk_weights"] = torch.ones((256, 1), device="cuda")
    calls = []
    native_gemm = implementation._native_gemm

    def tracked_native(a, weight, sa, sb, out, backend):
        calls.append((backend, tuple(a.shape), tuple(weight.shape)))
        return native_gemm(a, weight, sa, sb, out, backend)

    monkeypatch.setattr(implementation, "_native_gemm", tracked_native)
    actual = moe_ffn(**case, backend="auto")
    # 128 routes per expert satisfies the native threshold without padding
    # amplification. Both FFN projections must use the batched native path.
    assert calls == [
        ("cutlass", (2, 128, 256), (2, 512, 256)),
        ("cutlass", (2, 128, 256), (2, 256, 256)),
    ]
    expected = moe_ffn(**case, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.025)
    indices = [0, 127, 128, 255]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids"):
        subset[name] = case[name][indices].contiguous()
    assert_reference(actual[indices], subset)


@cuda
def test_thor_auto_falls_back_to_ragged_for_skewed_routing(monkeypatch):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Thor automatic native dispatch requires SM110")
    import grain_gemm.moe as implementation
    from grain_gemm.kernels import moe as kernels

    case = make_case(tokens=256, h=256, experts=3)
    case["topk_ids"] = torch.zeros((256, 2), device="cuda", dtype=torch.int64)
    case["topk_weights"] = torch.tensor([0.25, 1.5], device="cuda").repeat(256, 1)
    availability_calls, ragged_calls = [], []
    ragged_gemm = kernels.ragged_gemm

    def available(backend, device):
        availability_calls.append(backend)
        return True

    def unexpected_native(*args, **kwargs):
        pytest.fail("Skewed auto routing launched padded native expert GEMMs")

    def tracked_ragged(*args, **kwargs):
        ragged_calls.append(kwargs["source_is_token"])
        return ragged_gemm(*args, **kwargs)

    monkeypatch.setattr(implementation, "_native_available", available)
    monkeypatch.setattr(implementation, "_native_gemm", unexpected_native)
    monkeypatch.setattr(kernels, "ragged_gemm", tracked_ragged)
    actual = moe_ffn(**case, backend="auto")
    # 512 assignments first qualify for native (512 >= 3*128). Routing them
    # all to expert 0 would pad to 3*512 rows, exceeding 2*512 valid rows.
    assert availability_calls == ["cutlass"]
    assert ragged_calls == [True, False]
    expected = moe_ffn(**case, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    indices = [0, 127, 128, 255]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids"):
        subset[name] = case[name][indices].contiguous()
    assert_reference(actual[indices], subset)


@cuda
@pytest.mark.parametrize("field", ["hidden_states", "w_gate_up", "w_down", "scale_gate_up", "scale_down", "topk_weights"])
def test_zero_operand_gives_zero_output(field):
    case = make_case(tokens=3, h=256)
    case[field].zero_()
    actual = moe_ffn(**case)
    assert torch.count_nonzero(actual).item() == 0


@cuda
@pytest.mark.parametrize("value,rounded", [(0.5, 0), (1.5, 2), (2.5, 2), (-0.5, 0), (-1.5, -2), (-2.5, -2)])
def test_activation_quantization_uses_nearest_even(value, rounded):
    # amax=127 makes the input scale exactly one. The first expert projection
    # selects x[0] for the gate and x[1] for up, so an incorrect tie rule causes
    # a large, directly observable output difference rather than a small error.
    x = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    x[0, :2] = torch.tensor([127, value], device="cuda", dtype=x.dtype)
    first = torch.zeros((1, 512, 256), device="cuda", dtype=torch.int8)
    first[0, 0, 0] = 1
    first[0, 256, 1] = 1
    second = torch.zeros((1, 256, 256), device="cuda", dtype=torch.int8)
    second[0, :, 0] = 1
    result = moe_ffn(
        x, first.transpose(-2, -1), second.transpose(-2, -1),
        torch.ones((1, 1), device="cuda"),
        torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        scale_gate_up=torch.ones((1, 1, 512), device="cuda"),
        scale_down=torch.ones((1, 1, 256), device="cuda"),
        backend="triton",
    )
    torch.testing.assert_close(result, torch.full_like(result, 127 * rounded), rtol=0, atol=0)


@cuda
def test_repeated_experts_accumulate_each_slot():
    case = make_case(tokens=3, h=256)
    case["topk_ids"].fill_(1)
    case["topk_weights"][:] = torch.tensor([1.0, 2.0, -0.5], device="cuda")
    actual = moe_ffn(**case, backend="triton")
    assert_reference(actual, case)
    single = dict(case)
    single["topk_ids"] = case["topk_ids"][:, :1].contiguous()
    single["topk_weights"] = torch.full((3, 1), 2.5, device="cuda")
    torch.testing.assert_close(actual, moe_ffn(**single, backend="triton"), rtol=0, atol=0)


@cuda
def test_int32_expert_indices_and_empty_tokens():
    case = make_case(tokens=3, h=256)
    case["topk_ids"] = case["topk_ids"].int()
    assert_reference(moe_ffn(**case), case)
    for name in ("hidden_states", "topk_weights", "topk_ids"):
        case[name] = case[name][:0]
    empty = moe_ffn(**case)
    assert empty.shape == (0, 256)
    assert empty.dtype == case["hidden_states"].dtype
    assert empty.device == case["hidden_states"].device


@cuda
def test_all_masked_tokens_are_zero():
    case = make_case(tokens=3, h=256)
    case["token_mask"] = torch.zeros(3, dtype=torch.bool, device="cuda")
    actual = moe_ffn(**case)
    assert torch.count_nonzero(actual).item() == 0


@cuda
def test_nondefault_stream():
    case = make_case(tokens=3, h=256)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = moe_ffn(**case, backend="auto")
    torch.cuda.current_stream().wait_stream(stream)
    assert_reference(actual, case)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton", "cuda", "cutlass"])
def test_graph_replay_consumes_activation_mask_scores_and_routes(backend):
    require_backend(backend, torch.bfloat16)
    case = make_case(tokens=3, h=256)
    case["token_mask"] = torch.ones(3, dtype=torch.bool, device="cuda")
    # Native capture reserves round_up(T*R,64)=64 rows per expert. All nine
    # assignments can move to a previously empty expert without truncation.
    # Auto captures Triton so it needs no host count reads or padded capacity.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            moe_ffn(**case, backend="triton" if backend == "auto" else backend)
    stream.synchronize()
    original = {name: value.clone() for name, value in case.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = moe_ffn(**case, backend=backend)
    graph.replay()
    assert_reference(captured, case)
    case["hidden_states"].zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    case["hidden_states"].copy_(original["hidden_states"])
    graph.replay()
    assert_reference(captured, case)
    # Change the per-expert token counts while replaying the same graph.
    case["topk_ids"].fill_(4)
    case["topk_weights"][:] = torch.tensor([0.5, 0.25, 1.25], device="cuda")
    graph.replay()
    assert_reference(captured, case)
    case["token_mask"].zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    for name in ("topk_ids", "topk_weights", "token_mask"):
        case[name].copy_(original[name])
    graph.replay()
    assert_reference(captured, case)


@cuda
@pytest.mark.parametrize("config_id", [32, 33])
def test_thor_batched_gemm_preserves_expert_and_group_scale_strides(config_id):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Direct batched expert GEMM requires Thor SM110")
    require_backend("cutlass", torch.bfloat16)
    from grain_gemm.kernels import cutlass

    experts, m, n, k = 3, 192, 320, 768
    groups = k // 256
    generator = torch.Generator().manual_seed(2026)
    a = torch.randint(-8, 9, (experts, m, k), generator=generator, dtype=torch.int8)
    weight = torch.randint(-8, 9, (experts, n, k), generator=generator, dtype=torch.int8)
    expert = torch.arange(experts)[:, None, None]
    row = torch.arange(m)[None, :, None]
    column = torch.arange(n)[None, None, :]
    group = torch.arange(groups)
    # Small binary fractions keep all products and group sums exactly
    # representable in FP32, including cancellation, signed and zero scales.
    sa = (((expert * 3 + row * 5 + group[None, None, :] * 7) % 9) - 4).float() / 32
    sb = (((expert * 5 + group[None, :, None] * 3 + column * 7) % 11) - 5).float() / 64
    expected = torch.stack([
        group_dot_cpu(a[e], weight[e].t(), sa[e], sb[e]).bfloat16()
        for e in range(experts)
    ])
    a, weight, sa, sb = (tensor.cuda() for tensor in (a, weight, sa, sb))
    out = torch.empty((experts, m, n), device="cuda", dtype=torch.bfloat16)
    assert a.stride() == (m * k, k, 1)
    assert weight.stride() == (n * k, k, 1)
    assert sa.stride() == (m * groups, groups, 1)
    assert sb.stride() == (groups * n, n, 1)
    assert out.stride() == (m * n, n, 1)
    cutlass.launch_batched(a, weight, sa, sb, out, config_id=config_id)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("field", ["w_gate_up", "w_down"])
def test_contiguous_right_multiply_weight_is_rejected(field):
    case = make_case(tokens=2, h=256)
    case[field] = case[field].contiguous()
    with pytest.raises(ValueError):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize("field", ["hidden_states", "scale_gate_up", "scale_down", "topk_weights", "topk_ids"])
def test_noncanonical_stride_is_rejected(field):
    case = make_case(tokens=3, h=256)
    old = case[field]
    backing = torch.empty((*old.shape[:-1], old.shape[-1] * 2), device="cuda", dtype=old.dtype)
    view = backing[..., ::2]
    view.copy_(old)
    case[field] = view
    with pytest.raises(ValueError):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize(
    "field,dtype",
    [
        ("hidden_states", torch.float32),
        ("w_gate_up", torch.bfloat16),
        ("w_down", torch.bfloat16),
        ("scale_gate_up", torch.float16),
        ("scale_down", torch.float64),
        ("topk_weights", torch.bfloat16),
        ("topk_ids", torch.int16),
    ],
)
def test_invalid_dtype(field, dtype):
    case = make_case(tokens=2, h=256)
    case[field] = case[field].to(dtype)
    with pytest.raises((TypeError, ValueError)):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize("field", ["w_gate_up", "w_down", "scale_gate_up", "scale_down", "topk_weights", "topk_ids"])
def test_mismatched_shape(field):
    case = make_case(tokens=3, h=256)
    case[field] = case[field][:-1]
    with pytest.raises(ValueError):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize("bad_id", [-1, 5])
def test_out_of_range_expert_id_is_rejected_eagerly(bad_id):
    case = make_case(tokens=2, h=256)
    case["topk_ids"][0, 0] = bad_id
    with pytest.raises((ValueError, RuntimeError)):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize("argument,value", [("group_size", 128), ("group_size", True), ("activation", "gelu"), ("backend", "unknown")])
def test_unsupported_configuration_is_rejected(argument, value):
    with pytest.raises(ValueError):
        moe_ffn(**make_case(tokens=2, h=256), **{argument: value})


@cuda
@pytest.mark.parametrize("backend", ["cuda", "cutlass"])
def test_explicit_native_backend_never_silently_falls_back(monkeypatch, backend):
    import grain_gemm.moe as implementation

    monkeypatch.setattr(implementation, "_native_available", lambda *args: False)
    with pytest.raises(RuntimeError, match="native"):
        moe_ffn(**make_case(tokens=2, h=256), backend=backend)


@cuda
@pytest.mark.parametrize("backend", ["cuda", "cutlass"])
def test_native_backends_reject_fp16(backend):
    with pytest.raises(ValueError, match="BF16"):
        moe_ffn(**make_case(tokens=2, h=256, dtype=torch.float16), backend=backend)


@cuda
def test_cpu_argument_is_rejected():
    case = make_case(tokens=2, h=256)
    case["scale_down"] = case["scale_down"].cpu()
    with pytest.raises(ValueError):
        moe_ffn(**case)


@cuda
def test_token_mask_contract():
    case = make_case(tokens=3, h=256)
    for mask in (
        torch.ones(3, device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.bool),
        torch.ones(6, device="cuda", dtype=torch.bool)[::2],
    ):
        with pytest.raises((TypeError, ValueError)):
            moe_ffn(**case, token_mask=mask)
