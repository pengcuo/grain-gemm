"""MoE FFN contracts and independent, per-token CPU INT32 references.

The reference deliberately does not sort, pad, or pack tokens by expert. Each
top-k slot is evaluated separately, including repeated expert IDs.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import moe_ffn
from grain_gemm.kernels import cutlass_moe


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


def test_thor_native_selector_retains_measured_projection_choices():
    """Large expert matrices keep the original tile without broadening tuning."""
    from grain_gemm.moe import _native_config

    for n, k in ((2048, 1536), (1536, 1024)):
        for capacity in (1024, 2048, 4096):
            assert _native_config(capacity, n, k, experts=32) == 47
        assert _native_config(320, n, k, experts=32) == 44
        assert _native_config(640, n, k, experts=32) == (44 if n == 1536 else 43)
        assert _native_config(896, n, k, experts=32) == 43
        assert _native_config(1024, n, k, experts=16) == 43
    assert _native_config(1152, 2048, 1536, experts=32) == 47
    assert _native_config(1152, 1536, 1024, experts=32) == 44
    assert _native_config(1088, 1536, 1024, experts=32) == 43
    assert _native_config(2176, 1536, 1024, experts=32) == 43
    assert _native_config(2112, 2048, 1536, experts=32) == 47
    assert _native_config(2112, 1536, 1024, experts=32) == 43
    assert _native_config(1280, 2048, 1536, experts=32) == 43
    assert _native_config(1024, 1024, 1536, experts=32) == 43
    assert _native_config(1024, 2048, 1024, experts=32) == 43


def routing_fixture_cpu():
    ids = torch.tensor([
        [0, 2, 2],
        [2**40, 1, -(2**40)],
        [2**63 - 1, -(2**63), 4],
        [2, 0, 1],
    ], dtype=torch.int64)
    mask = torch.tensor([True, True, False, True])
    return ids, mask


@cuda
@pytest.mark.parametrize("include_tiles", [False, True])
@pytest.mark.parametrize("kind", ["mixed", "empty", "all_masked", "multi_block"])
def test_route_statistics_match_independent_counts(kind, include_tiles):
    from grain_gemm.kernels import moe as kernels

    ids, mask = routing_fixture_cpu()
    if kind == "empty":
        ids, mask = ids[:0], mask[:0]
    elif kind == "all_masked":
        mask.zero_()
    elif kind == "multi_block":
        # Invalid IDs straddle the 1024-assignment kernel boundary. The middle
        # one is masked, so the prefix reduction must sum 1+1 invalid routes.
        ids = (torch.arange(1031) % 5).reshape(-1, 1)
        ids[0, 0], ids[1024, 0], ids[1030, 0] = 2**40, -(2**40), 2**63 - 1
        mask = torch.ones(1031, dtype=torch.bool)
        mask[1024] = False
    experts = 5
    counts = [0] * experts
    keys, invalid = [], 0
    for token, choices in enumerate(ids.tolist()):
        for expert in choices:
            valid = bool(mask[token]) and 0 <= expert < experts
            keys.append(expert if valid else experts)
            if valid:
                counts[expert] += 1
            elif mask[token]:
                invalid += 1
    total = sum(counts)
    offsets = [sum(counts[:expert]) for expert in range(experts + 1)]
    valid_order = sorted(
        (assignment for assignment, expert in enumerate(keys) if expert < experts),
        key=keys.__getitem__,
    )
    inverse = [-1] * ids.numel()
    for slot, assignment in enumerate(valid_order):
        inverse[assignment] = slot
    routes = kernels.prepare_routes(ids.cuda(), experts, mask.cuda(), include_tiles=include_tiles)
    assert routes.statistics.dtype == torch.int32 and routes.statistics.shape == (3,)
    assert routes.statistics.cpu().tolist() == [invalid, max(counts), total]
    assert routes.counts.cpu().tolist() == counts
    assert routes.offsets.cpu().tolist() == offsets
    assert routes.effective_ids.cpu().tolist() == keys
    assert routes.order[:total].cpu().tolist() == valid_order
    assert routes.inverse.cpu().tolist() == inverse
    if not include_tiles:
        assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
        return
    assert routes.tile_offsets is not None
    assert routes.tile_experts is not None
    assert routes.tile_starts is not None
    tile_counts = [(count + routes.block_m - 1) // routes.block_m for count in counts]
    assert routes.tile_offsets.cpu().tolist() == [
        sum(tile_counts[:expert]) for expert in range(experts + 1)
    ]
    expected_experts, expected_starts = [], []
    for expert, count in enumerate(tile_counts):
        for tile in range(count):
            expected_experts.append(expert)
            expected_starts.append(offsets[expert] + tile * routes.block_m)
    active_tiles = sum(tile_counts)
    assert routes.tile_experts[:active_tiles].cpu().tolist() == expected_experts
    assert routes.tile_starts[:active_tiles].cpu().tolist() == expected_starts
    assert (routes.tile_experts[active_tiles:] < 0).all().item()


@cuda
@pytest.mark.parametrize("experts,tokens", [(32, 1031), (64, 1031), (65, 1031),
                                           (64, 65536), (64, 65537)])
def test_multichunk_routes_preserve_stable_order_and_invalid_tail(experts, tokens):
    """Chunk boundaries and the small-expert boundary preserve every slot."""
    from grain_gemm.kernels import moe as kernels

    ids = ((torch.arange(tokens, dtype=torch.int64) * 17 + 11) % experts).reshape(-1, 1)
    mask = torch.arange(tokens) % 11 != 0
    for token, value in ((1, 2**40), (255, -(2**40)), (256, experts),
                         (1024, 2**63 - 1), (1030, -(2**63))):
        ids[token, 0] = value
    counts = [0] * experts
    keys, invalid = [], 0
    for token, (expert,) in enumerate(ids.tolist()):
        enabled = bool(mask[token])
        valid = enabled and 0 <= expert < experts
        keys.append(expert if valid else experts)
        if valid:
            counts[expert] += 1
        elif enabled:
            invalid += 1
    order = sorted(range(tokens), key=keys.__getitem__)
    inverse = [-1] * tokens
    for position, assignment in enumerate(order):
        if keys[assignment] < experts:
            inverse[assignment] = position
    offsets = [sum(counts[:expert]) for expert in range(experts + 1)]

    routes = kernels.prepare_routes(ids.cuda(), experts, mask.cuda(), include_tiles=False)
    assert routes.effective_ids.cpu().tolist() == keys
    # Include the invalid tail: it is stable too, although downstream GEMMs
    # only consume the valid prefix and inverse=-1 for invalid assignments.
    assert routes.order.cpu().tolist() == order
    assert routes.inverse.cpu().tolist() == inverse
    assert routes.counts.cpu().tolist() == counts
    assert routes.offsets.cpu().tolist() == offsets
    assert routes.statistics.cpu().tolist() == [invalid, max(counts), sum(counts)]


@cuda
def test_routes_without_tiles_support_pack_combine_and_late_tile_preparation():
    from grain_gemm.kernels import moe as kernels

    ids, mask = routing_fixture_cpu()
    experts, capacity, k, output_width = 5, 64, 256, 16
    routes = kernels.prepare_routes(ids.cuda(), experts, mask.cuda(), include_tiles=False)
    assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
    q = (torch.arange(ids.shape[0] * k).reshape(ids.shape[0], k) % 17 - 8).to(torch.int8)
    sa = torch.arange(1, ids.shape[0] + 1, dtype=torch.float32).reshape(-1, 1)
    scores = torch.tensor([
        [0.25, 1.5, -0.5], [2.0, 0.5, 1.0],
        [0.125, 3.0, 0.25], [1.5, -0.25, 0.75],
    ])
    expected_q = torch.zeros((experts, capacity, k), dtype=torch.int8)
    expected_sa = torch.zeros((experts, capacity, 1), dtype=torch.float32)
    expected_output = torch.zeros((ids.shape[0], output_width), dtype=torch.float32)
    counts = [0] * experts
    for token, choices in enumerate(ids.tolist()):
        if not mask[token]:
            continue
        for slot, expert in enumerate(choices):
            if not 0 <= expert < experts:
                continue
            local_row = counts[expert]
            expected_q[expert, local_row] = q[token]
            expected_sa[expert, local_row] = sa[token]
            counts[expert] += 1
            expected_output[token] += (
                q[token, :output_width].float() + expert * 16
            ) * scores[token, slot]
    packed_q, packed_sa = kernels.pack_inputs(q.cuda(), sa.cuda(), routes, capacity)
    torch.testing.assert_close(packed_q.cpu(), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(packed_sa.cpu(), expected_sa, rtol=0, atol=0)
    # Padding receives nonzero artificial outputs too: combine must rely on
    # routing validity, including masked and INT64 out-of-range assignments.
    expert_bias = torch.arange(experts, device="cuda")[:, None, None] * 16
    expert_output = (packed_q[:, :, :output_width].float() + expert_bias).bfloat16().contiguous()
    actual = kernels.combine(
        expert_output, scores.cuda(), routes, output_dtype=torch.bfloat16,
        packed_capacity=capacity,
    )
    torch.testing.assert_close(actual.cpu(), expected_output.bfloat16(), rtol=0, atol=0)
    completed = kernels.prepare_ragged_tiles(routes)
    assert completed is not routes
    assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
    assert completed.tile_offsets is not None
    assert completed.tile_experts is not None
    assert completed.tile_starts is not None
    assert completed.counts is routes.counts and completed.statistics is routes.statistics
    assert kernels.prepare_ragged_tiles(completed) is completed
    torch.testing.assert_close(
        kernels.combine(expert_output, scores.cuda(), completed,
                        output_dtype=torch.bfloat16, packed_capacity=capacity),
        actual, rtol=0, atol=0,
    )


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capacity", [17, 64])
@pytest.mark.parametrize("zero_initialize", [False, True], ids=["padding_fill", "full_fill"])
def test_quantize_pack_matches_independent_routes_and_unfused_bits(dtype, capacity, zero_initialize):
    """Fused quantization preserves duplicate routes and fully defines padding."""
    from grain_gemm.kernels import moe as kernels

    tokens, experts, top_k, k = 19, 5, 3, 768
    generator = torch.Generator().manual_seed(20261020)
    x = torch.randn((tokens, k), generator=generator).to(dtype)
    # Exact unit scales expose nearest-even ties in every reduction group;
    # another valid row exercises the clamp_min scale for an all-zero group.
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
    expected_q = torch.zeros((experts, capacity, k), dtype=torch.int8)
    expected_s = torch.zeros((experts, capacity, k // 256), dtype=torch.float32)
    counts = [0] * experts
    for token in range(tokens):
        if not mask[token]:
            continue
        q, scale = quantize_rows_cpu(x[token : token + 1])
        for expert in ids[token].tolist():
            row = counts[expert]
            expected_q[expert, row] = q[0]
            expected_s[expert, row] = scale[0]
            counts[expert] += 1
    assert counts == [16, 15, 17, 0, 0]

    def offset_cuda(value):
        backing = torch.empty(value.numel() + 16, device="cuda", dtype=value.dtype)
        view = backing[16:].view(value.shape)
        view.copy_(value)
        assert view.storage_offset() == 16 and view.data_ptr() % 16 == 0
        return view

    x, ids, mask = (offset_cuda(value) for value in (x, ids, mask))
    routes = kernels.prepare_routes(ids, experts, mask, include_tiles=False)
    qx, sx = kernels.quantize_rows(x)
    unfused_q, unfused_s = kernels.pack_inputs(qx, sx, routes, capacity)
    actual_q, actual_s = kernels.quantize_pack_inputs(
        x, routes, capacity, zero_initialize=zero_initialize,
    )
    for actual in (unfused_q.cpu(), actual_q.cpu()):
        torch.testing.assert_close(actual, expected_q, rtol=0, atol=0)
    for actual in (unfused_s.cpu(), actual_s.cpu()):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.view(torch.int32), expected_s.view(torch.int32), rtol=0, atol=0)


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
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_masked_nan_offset_views_and_signed_scales_preserve_ffn(backend, dtype):
    """Optimized dispatch retains public mask, view and numerical behavior."""
    case = make_case(tokens=129, h=512, i=256, experts=4, dtype=dtype)
    case["topk_ids"] = torch.arange(4, device="cuda").repeat(129, 1)
    case["topk_ids"][64].fill_(1)
    case["topk_weights"] = torch.tensor([0.25, 1.5, -0.5, 0.0], device="cuda").repeat(129, 1)
    case["token_mask"] = torch.ones(129, dtype=torch.bool, device="cuda")
    masked_rows = [0, 63, 128]
    case["token_mask"][masked_rows] = False
    case["hidden_states"][masked_rows] = float("nan")
    case["topk_weights"][masked_rows] = float("nan")
    case["topk_ids"][masked_rows] = torch.tensor(
        [2**40, -(2**40), 2**63 - 1, -(2**63)], device="cuda", dtype=torch.int64
    )
    for name in ("scale_gate_up", "scale_down"):
        case[name][:, ::2, ::3].mul_(-1)
        case[name][..., 5::17].zero_()
    # Every view keeps its required shape/stride and 16-byte alignment, while
    # its tensor pointer differs from the storage base. Native descriptors and
    # fused gathers must consume the view's data_ptr and honor masked rows.
    for name, value in case.items():
        backing = torch.empty(value.numel() + 16, device=value.device, dtype=value.dtype)
        offset_view = torch.as_strided(backing, value.shape, value.stride(), storage_offset=16)
        offset_view.copy_(value)
        assert offset_view.storage_offset() == 16
        case[name] = offset_view
    actual = moe_ffn(**case, backend=backend)
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual[masked_rows], torch.zeros_like(actual[masked_rows]), rtol=0, atol=0)
    # This shape qualifies for Thor's eager native auto path. A few rows retain
    # the full H/I reduction and all routes, including duplicate expert slots,
    # without making an independent INT32 CPU reference unnecessarily large.
    indices = [0, 1, 63, 64, 127, 128]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids", "token_mask"):
        subset[name] = case[name][indices].contiguous()
    assert_reference(actual[indices], subset)


@cuda
@pytest.mark.parametrize("backend", ["auto", "cuda", "cutlass"])
def test_thor_auto_selects_native_for_balanced_routing(monkeypatch, backend):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Thor automatic native dispatch requires SM110")
    selected = "cutlass" if backend == "auto" else backend
    require_backend(selected, torch.bfloat16)
    import grain_gemm.moe as implementation
    from grain_gemm.kernels import moe as kernels

    case = make_case(tokens=256, h=256, experts=2)
    case["topk_ids"] = (torch.arange(256, device="cuda") % 2).reshape(256, 1)
    case["topk_weights"] = torch.ones((256, 1), device="cuda")
    calls, route_calls = [], []
    native_gemm = implementation._native_gemm
    prepare_routes = kernels.prepare_routes

    def tracked_native(a, weight, sa, sb, out, backend, *, counts=None):
        calls.append((backend, tuple(a.shape), tuple(weight.shape)))
        return native_gemm(a, weight, sa, sb, out, backend, counts=counts)

    def tracked_routes(*args, **kwargs):
        route_calls.append(kwargs.get("include_tiles", True))
        routes = prepare_routes(*args, **kwargs)
        assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
        return routes

    def unexpected_tiles(*args, **kwargs):
        pytest.fail("Native expert GEMMs prepared unused ragged tile metadata")

    with monkeypatch.context() as native_patch:
        native_patch.setattr(implementation, "_native_gemm", tracked_native)
        native_patch.setattr(kernels, "prepare_routes", tracked_routes)
        native_patch.setattr(kernels, "prepare_ragged_tiles", unexpected_tiles)
        actual = moe_ffn(**case, backend=backend)
    # 128 routes per expert satisfies the native threshold without padding
    # amplification. Both FFN projections must use the batched native path.
    assert calls == [
        (selected, (2, 128, 256), (2, 512, 256)),
        (selected, (2, 128, 256), (2, 256, 256)),
    ]
    assert route_calls == [False]
    expected = moe_ffn(**case, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.025)
    indices = [0, 127, 128, 255]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids"):
        subset[name] = case[name][indices].contiguous()
    assert_reference(actual[indices], subset)


@cuda
def test_thor_auto_uses_original_tile_for_large_expert_ffn(monkeypatch):
    """The native fallback retains both measured tiles and complete FFN math."""
    if torch.cuda.get_device_capability() != (11, 0) or not cutlass_moe.is_available():
        pytest.skip("Build the dedicated Thor MoE library on SM110")
    import grain_gemm.moe as implementation
    # Check the existing native fallback even when the optional fused G1 build
    # is installed; its automatic selection has separate public API coverage.
    monkeypatch.setattr(implementation, "_use_fused_gate_up", lambda *args, **kwargs: False)

    tokens, experts, top_k = 8192, 32, 4
    case = make_case(tokens=tokens, experts=experts, h=1536, i=1024)
    case["topk_ids"] = (
        torch.arange(tokens * top_k, device="cuda", dtype=torch.int64)
        .remainder(experts).reshape(tokens, top_k)
    )
    case["topk_weights"] = torch.tensor(
        [0.25, -0.5, 1.5, 0.0], device="cuda", dtype=torch.float32,
    ).repeat(tokens, 1)
    calls = []
    native_launch = cutlass_moe._launch_prevalidated

    def tracked_launch(a, weight, sa, sb, out, config_id=34, **kwargs):
        calls.append((config_id, tuple(a.shape), tuple(weight.shape)))
        return native_launch(a, weight, sa, sb, out, config_id, **kwargs)

    with monkeypatch.context() as native_patch:
        native_patch.setattr(cutlass_moe, "_launch_prevalidated", tracked_launch)
        actual = moe_ffn(**case, backend="auto")
    assert calls == [
        (47, (32, 1024, 1536), (32, 2048, 1536)),
        (47, (32, 1024, 1024), (32, 1536, 1024)),
    ]
    calls.clear()
    with monkeypatch.context() as native_patch:
        native_patch.setattr(implementation, "_native_config", lambda *args, **kwargs: 34)
        native_patch.setattr(cutlass_moe, "_launch_prevalidated", tracked_launch)
        expected = moe_ffn(**case, backend="auto")
    assert calls == [
        (34, (32, 1024, 1536), (32, 2048, 1536)),
        (34, (32, 1024, 1024), (32, 1536, 1024)),
    ]
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
    indices = [0, 3, 4094, tokens - 1]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids"):
        subset[name] = case[name][indices].contiguous()
    assert_reference(actual[indices], subset)


@cuda
@pytest.mark.parametrize("config_id", [43, 44, 46, 47])
def test_thor_counted_ffn_ignores_unwritten_nan_padding(monkeypatch, config_id):
    """Sparse native padding cannot reach SwiGLU or valid combined outputs."""
    if torch.cuda.get_device_capability() != (11, 0) or not cutlass_moe.is_available():
        pytest.skip("Build the dedicated Thor MoE library on SM110")
    import grain_gemm.moe as implementation

    case = make_case(tokens=257, h=256, i=256, experts=4)
    case["topk_ids"] = torch.stack(
        (torch.zeros(257, dtype=torch.int64), torch.arange(257) % 3), dim=1,
    ).cuda()
    case["topk_weights"] = torch.tensor([0.25, -0.5], device="cuda").repeat(257, 1)
    case["token_mask"] = torch.ones(257, dtype=torch.bool, device="cuda")
    masked_rows = [0, 128, 256]
    case["token_mask"][masked_rows] = False
    case["hidden_states"][masked_rows] = float("nan")
    case["topk_weights"][masked_rows] = float("nan")
    case["topk_ids"][masked_rows] = torch.tensor([2**63 - 1, -(2**40)], device="cuda")
    expected_counts = [0] * 4
    for token in range(257):
        if token not in masked_rows:
            expected_counts[0] += 1
            expected_counts[token % 3] += 1
    assert max(expected_counts) > 256 and expected_counts[3] == 0
    calls = []
    native = implementation._native_gemm

    def poison_padding(a, weight, sa, sb, out, backend, *, counts=None):
        assert counts is not None
        assert counts.cpu().tolist() == expected_counts
        # On FC2 this also inspects padded hidden scales: an unmasked SwiGLU
        # load from the poisoned FC1 padding would leave NaNs here even if
        # the final route combination happened to ignore those rows.
        assert torch.isfinite(sa).all().item()
        config = implementation._native_config(a.shape[1], weight.shape[1], a.shape[2])
        calls.append((config, a.shape[1]))
        # A counts-aware scheduler may leave whole padding tiles untouched.
        # Start both native outputs as NaNs so ignoring the live-row mask in
        # SwiGLU or combination becomes an observable correctness failure.
        out.fill_(float("nan"))
        return native(a, weight, sa, sb, out, backend, counts=counts)

    with monkeypatch.context() as native_patch:
        # Exercise the counted scheduler with a small independent-reference
        # fixture; production tuning uses larger E32 projection shapes.
        native_patch.setattr(implementation, "_native_config", lambda *args, **kwargs: config_id)
        native_patch.setattr(implementation, "_native_gemm", poison_padding)
        actual = moe_ffn(**case, backend="cutlass")
    assert calls == [(config_id, 384), (config_id, 384)]
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual[masked_rows], torch.zeros_like(actual[masked_rows]), rtol=0, atol=0)
    indices = [0, 1, 2, 127, 255, 256]
    subset = dict(case)
    for name in ("hidden_states", "topk_weights", "topk_ids", "token_mask"):
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
    availability_calls, ragged_calls, route_calls, tile_calls = [], [], [], []
    ragged_gemm = kernels.ragged_gemm
    prepare_routes = kernels.prepare_routes
    prepare_tiles = kernels.prepare_ragged_tiles

    def available(backend, device):
        availability_calls.append(backend)
        return True

    def unexpected_native(*args, **kwargs):
        pytest.fail("Skewed auto routing launched padded native expert GEMMs")

    def tracked_ragged(*args, **kwargs):
        ragged_calls.append(kwargs["source_is_token"])
        routes = args[4]
        assert routes.tile_offsets is not None
        assert routes.tile_experts is not None
        assert routes.tile_starts is not None
        return ragged_gemm(*args, **kwargs)

    def tracked_routes(*args, **kwargs):
        route_calls.append(kwargs.get("include_tiles", True))
        routes = prepare_routes(*args, **kwargs)
        assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
        return routes

    def tracked_tiles(routes):
        assert routes.tile_offsets is routes.tile_experts is routes.tile_starts is None
        tile_calls.append(routes)
        result = prepare_tiles(routes)
        assert result is not routes
        return result

    with monkeypatch.context() as routing_patch:
        routing_patch.setattr(implementation, "_native_available", available)
        routing_patch.setattr(implementation, "_native_gemm", unexpected_native)
        routing_patch.setattr(kernels, "ragged_gemm", tracked_ragged)
        routing_patch.setattr(kernels, "prepare_routes", tracked_routes)
        routing_patch.setattr(kernels, "prepare_ragged_tiles", tracked_tiles)
        actual = moe_ffn(**case, backend="auto")
    # 512 assignments first qualify for native (512 >= 3*128). Routing them
    # all to expert 0 would pad to 3*512 rows, exceeding 2*512 valid rows.
    assert availability_calls == ["cutlass"]
    assert ragged_calls == [True, False]
    assert route_calls == [False]
    assert len(tile_calls) == 1
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


def native_moe_operands_cpu(experts, m, n, k):
    """Integer extrema and signed binary scales with exactly FP32 group sums."""
    groups = k // 256
    generator = torch.Generator().manual_seed(20261020)
    a = torch.randint(-8, 9, (experts, m, k), generator=generator, dtype=torch.int8)
    weight = torch.randint(-8, 9, (experts, n, k), generator=generator, dtype=torch.int8)
    for start in range(0, k, 256):
        a[..., start], a[..., start + 1] = -128, 127
        weight[..., start], weight[..., start + 1] = 127, -128
    expert = torch.arange(experts)[:, None, None]
    row = torch.arange(m)[None, :, None]
    column = torch.arange(n)[None, None, :]
    group = torch.arange(groups)
    # Binary fractions make products and the full group sums exactly FP32
    # representable, including zero scales and cancellation across groups.
    sa = (((expert * 3 + row * 5 + group[None, None, :] * 7) % 9) - 4).float() / 32
    sb = (((expert * 5 + group[None, :, None] * 3 + column * 7) % 11) - 5).float() / 64
    return a, weight, sa, sb


@cuda
@pytest.mark.parametrize("config_id", tuple(cutlass_moe.CONFIGS))
@pytest.mark.parametrize("experts", [1, 3], ids=["single_expert", "batched_experts"])
def test_thor_moe_tiles_preserve_group_math_and_offset_views(config_id, experts):
    """New 1SM/2SM tiles retain exact G256 math at partial M/N boundaries."""
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("The dedicated MoE collective requires Thor SM110")
    if not cutlass_moe.is_available():
        pytest.skip("Build the optional Thor MoE library")
    assert config_id in cutlass_moe.CONFIGS
    m, n, k = 192, 320, 768
    a, weight, sa, sb = native_moe_operands_cpu(experts, m, n, k)
    rows = torch.linspace(0, m - 1, 16).long()
    columns = torch.linspace(0, n - 1, 16).long()
    expected = torch.stack([
        group_dot_cpu(
            a[e].index_select(0, rows), weight[e].index_select(0, columns).t(),
            sa[e].index_select(0, rows), sb[e].index_select(1, columns),
        ).bfloat16()
        for e in range(experts)
    ])

    def cuda_offset_view(value, elements):
        backing = torch.empty(value.numel() + elements, device="cuda", dtype=value.dtype)
        view = backing[elements:].view(value.shape)
        view.copy_(value)
        assert view.storage_offset() == elements
        return view

    a = cuda_offset_view(a, 16)
    weight = cuda_offset_view(weight, 16)
    sa = cuda_offset_view(sa, 1)  # Valid scalar SFA alignment, deliberately not 16-byte aligned.
    sb = cuda_offset_view(sb, 4)
    out = cuda_offset_view(torch.full((experts, m, n), float("nan"), dtype=torch.bfloat16), 8)
    assert sa.data_ptr() % 16 == 4
    assert all(tensor.data_ptr() % 16 == 0 for tensor in (a, weight, sb, out))
    cutlass_moe.launch(a, weight, sa, sb, out, config_id=config_id)
    actual = out.cpu()
    assert torch.isfinite(actual).all()
    actual_sample = actual.index_select(1, rows).index_select(2, columns)
    torch.testing.assert_close(actual_sample.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("config_id", tuple(
    config for config, metadata in cutlass_moe.CONFIGS.items() if metadata.get("counts_aware")
))
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph_replay"])
def test_thor_moe_counted_scheduler_consumes_uneven_and_changing_populations(config_id, captured):
    """Device counts retain valid rows across empty experts and tile tails."""
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("The dedicated MoE collective requires Thor SM110")
    if not cutlass_moe.is_available():
        pytest.skip("Build the optional Thor MoE library")
    experts, capacity, n, k = 6, 320, 320, 768
    host_operands = native_moe_operands_cpu(experts, capacity, n, k)
    operands = [value.cuda() for value in host_operands]
    out = torch.empty((experts, capacity, n), dtype=torch.bfloat16, device="cuda")
    # A four-byte offset is a valid count pointer. The scheduler must consume
    # the tensor view, including on replay, rather than its allocation base.
    count_storage = torch.full((experts + 1,), -777, dtype=torch.int32, device="cuda")
    counts = count_storage[1:]
    assert counts.storage_offset() == 1 and counts.data_ptr() % 4 == 0
    populations = [[0, 1, 127, 128, 129, 319]]
    if captured:
        populations += [[0] * experts, [320, 0, 129, 1, 0, 128], [1, 319, 128, 127, 0, 320]]
    counts.copy_(torch.tensor(populations[0], dtype=torch.int32))

    def run():
        return cutlass_moe.launch(*operands, out, config_id=config_id, counts=counts)

    if captured:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()

    columns = torch.linspace(0, n - 1, 16).long()
    for population in populations:
        counts.copy_(torch.tensor(population, dtype=torch.int32))
        out.fill_(float("nan"))
        if captured:
            graph.replay()
        else:
            run()
        actual = out.cpu()
        for expert, count in enumerate(population):
            # Whole padding clusters must not write even their first scheduled
            # tile. NaN poison directly detects a missing initial-work skip.
            cluster_rows = cutlass_moe.CONFIGS[config_id]["block_m"]
            padding_begin = ((count + cluster_rows - 1) // cluster_rows) * cluster_rows
            assert torch.isnan(actual[expert, padding_begin:]).all()
            if count == 0:
                continue
            assert torch.isfinite(actual[expert, :count]).all()
            rows = torch.linspace(0, count - 1, min(count, 16)).long().unique()
            a, weight, sa, sb = host_operands
            expected = group_dot_cpu(
                a[expert].index_select(0, rows), weight[expert].index_select(0, columns).t(),
                sa[expert].index_select(0, rows), sb[expert].index_select(1, columns),
            ).bfloat16()
            sampled = actual[expert].index_select(0, rows).index_select(1, columns)
            torch.testing.assert_close(sampled.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
        # Padding rows may remain unwritten, including an empty expert's first
        # tile. Only [0, count_e) is part of this native scheduler's contract.


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
@pytest.mark.parametrize("bad_id", [-1, 5, 2**40, -(2**40), 2**63 - 1, -(2**63)])
def test_out_of_range_expert_id_is_rejected_eagerly(bad_id):
    case = make_case(tokens=2, h=256)
    case["topk_ids"][0, 0] = bad_id
    with pytest.raises(ValueError, match="topk_ids"):
        moe_ffn(**case)


@cuda
@pytest.mark.parametrize("bad_id", [2**40, -(2**40), 2**63 - 1, -(2**63)])
def test_masked_int64_out_of_range_expert_ids_are_ignored(bad_id):
    case = make_case(tokens=2, h=256)
    case["topk_ids"][0].fill_(bad_id)
    case["token_mask"] = torch.tensor([False, True], device="cuda")
    actual = moe_ffn(**case)
    assert torch.count_nonzero(actual[0]).item() == 0
    assert_reference(actual, case)


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
