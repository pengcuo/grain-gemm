"""Explicit gate/up channel ordering preserves the same mathematical FFN."""

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import moe_ffn
from grain_gemm import moe as implementation
from grain_gemm.kernels import moe as kernels, moe_gather, moe_interleaved


@pytest.fixture
def bridge_cache():
    moe_interleaved._load_library.cache_clear()
    yield
    moe_interleaved._load_library.cache_clear()


def _optional_build(tmp_path, monkeypatch, **overrides):
    library = tmp_path / "libgrain_moe_interleaved.so"
    library.write_bytes(b"test interleaved library")
    metadata = dict(architecture="sm_110", compiler_architecture="sm_110a",
                    abi_version=1, consumer_warps=[8, 16],
                    source_sha256=moe_interleaved.source_hashes(),
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    metadata.update(overrides)
    library.with_name("moe_interleaved_build.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(moe_interleaved, "_LIBRARY", library)
    return library


def _permutation(intermediate, device="cpu"):
    return torch.stack((torch.arange(intermediate, device=device),
                        torch.arange(intermediate, device=device) + intermediate), -1).reshape(-1)


def _interleave_case(case):
    result = dict(case)
    weight = case["w_gate_up"]
    order = _permutation(weight.shape[-1] // 2, weight.device)
    # Selecting Linear output rows preserves the required transposed view.
    result["w_gate_up"] = weight.transpose(-2, -1).index_select(1, order).transpose(-2, -1)
    result["scale_gate_up"] = case["scale_gate_up"].index_select(-1, order).contiguous()
    result["gate_up_layout"] = "interleaved"
    return result


def _require_cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("An SM80-or-newer CUDA device is required")


def _require_interleaved_native():
    _require_cuda()
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("The interleaved native first projection requires SM110")
    if not implementation._native_available("cutlass", "cuda", "interleaved"):
        pytest.skip("Build the optional interleaved MoE and native GEMM libraries")


def test_default_layout_stays_split_cpu():
    assert inspect.signature(moe_ffn).parameters["gate_up_layout"].default == "split"


@pytest.mark.parametrize("layout", ["unknown", "", None, True, 0])
def test_bad_layout_rejected_before_device_access_cpu(layout):
    with pytest.raises(ValueError, match="gate_up_layout"):
        moe_ffn(None, None, None, None, None, scale_x=None,
                scale_gate_up=None, scale_down=None, gate_up_layout=layout)
    with pytest.raises(ValueError, match="gate_up_layout"):
        kernels.swiglu_quantize(None, gate_up_layout=layout)


def test_weight_and_scale_permutation_preserves_group_scaled_projection_cpu():
    """Distinct per-group/per-output scales expose any missing SB permutation."""
    from test_moe_ffn import group_dot_cpu, make_case

    case = make_case(device="cpu", tokens=2, h=512, i=256)
    paired = _interleave_case(case)
    i = case["w_gate_up"].shape[-1] // 2
    assert paired["w_gate_up"].shape == case["w_gate_up"].shape
    assert paired["w_gate_up"].stride() == case["w_gate_up"].stride()
    assert paired["scale_gate_up"].stride() == case["scale_gate_up"].stride()
    split = group_dot_cpu(case["hidden_states"], case["w_gate_up"][0],
                          case["scale_x"], case["scale_gate_up"][0])
    actual = group_dot_cpu(case["hidden_states"], paired["w_gate_up"][0],
                           case["scale_x"], paired["scale_gate_up"][0])
    assert torch.equal(actual[:, 0::2], split[:, :i])
    assert torch.equal(actual[:, 1::2], split[:, i:])
    assert paired["w_down"] is case["w_down"]
    assert paired["scale_down"] is case["scale_down"]


@pytest.mark.parametrize("layout,available", [("split", False), ("interleaved", False),
                                               ("interleaved", True)])
def test_native_availability_requires_the_selected_layout_cpu(monkeypatch, layout, available):
    from grain_gemm import kernels as package
    from grain_gemm.kernels import cutlass_moe

    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (11, 0))
    monkeypatch.setattr(moe_gather, "is_available", lambda device: calls.append("split") or
                        (available if layout == "split" else False))
    monkeypatch.setattr(package, "moe_interleaved", SimpleNamespace(
        is_available=lambda device: calls.append("interleaved") or
        (available if layout == "interleaved" else False)), raising=False)
    monkeypatch.setattr(cutlass_moe, "is_available", lambda device: True)
    assert implementation._native_available("cutlass", "cuda:3", layout) is available
    assert calls == [layout]


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (9, 0), (11, 1)])
def test_other_architectures_do_not_load_interleaved_library_cpu(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(moe_interleaved, "_load_library",
                        lambda: pytest.fail("SM110 interleaved library loaded on another architecture"))
    assert not moe_interleaved.is_available("cuda:3")


def test_absent_interleaved_build_falls_back_cpu(tmp_path, monkeypatch, bridge_cache):
    monkeypatch.setattr(moe_interleaved, "_LIBRARY", tmp_path / "absent.so")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (11, 0))
    assert not moe_interleaved.is_available("cuda:3")


@pytest.mark.parametrize("overrides", [
    {"architecture": "sm_120"}, {"compiler_architecture": "sm_110"},
    {"abi_version": 0}, {"consumer_warps": [16]}, {"source_sha256": {}},
])
def test_stale_interleaved_build_rejected_before_loading_cpu(
        tmp_path, monkeypatch, bridge_cache, overrides):
    _optional_build(tmp_path, monkeypatch, **overrides)
    monkeypatch.setattr(moe_interleaved.ctypes, "CDLL", lambda path: pytest.fail("stale library loaded"))
    with pytest.raises(RuntimeError, match="stale or incompatible"):
        moe_interleaved._load_library()


def test_replaced_interleaved_binary_rejected_cpu(tmp_path, monkeypatch, bridge_cache):
    library = _optional_build(tmp_path, monkeypatch)
    library.write_bytes(b"different binary")
    monkeypatch.setattr(moe_interleaved.ctypes, "CDLL", lambda path: pytest.fail("replacement loaded"))
    with pytest.raises(RuntimeError, match="does not match"):
        moe_interleaved._load_library()


@pytest.mark.parametrize("contents", ["not json", "null", "[]"])
def test_bad_interleaved_manifest_rejected_cpu(tmp_path, monkeypatch, bridge_cache, contents):
    library = _optional_build(tmp_path, monkeypatch)
    library.with_name("moe_interleaved_build.json").write_text(contents)
    with pytest.raises(RuntimeError, match="metadata|incompatible"):
        moe_interleaved._load_library()


def test_interleaved_abi_uses_own_symbols_cpu(tmp_path, monkeypatch, bridge_cache):
    path = _optional_build(tmp_path, monkeypatch)
    library = SimpleNamespace(grain_moe_interleaved_launch=lambda *args: 0,
                              grain_moe_interleaved_smem_bytes=lambda warps: 167552,
                              grain_moe_interleaved_error_string=lambda code: b"error")
    loads = []
    monkeypatch.setattr(moe_interleaved.ctypes, "CDLL", lambda selected: loads.append(selected) or library)
    assert moe_interleaved._load_library() is library
    assert moe_interleaved._load_library() is library
    assert loads == [str(path)]
    assert library.grain_moe_interleaved_launch.argtypes == (
        [moe_interleaved.ctypes.c_void_p] * 9 + [moe_interleaved.ctypes.c_int] * 8
        + [moe_interleaved.ctypes.c_void_p])


def test_interleaved_launch_consumes_each_calls_pointers_and_stream_cpu(monkeypatch):
    calls = []
    streams = iter([71, 83])
    device = object()

    class Tensor:
        def __init__(self, ptr, shape=()):
            self.ptr, self.shape, self.device = ptr, shape, device

        def data_ptr(self):
            return self.ptr

    def current_stream(selected):
        assert selected is device
        return SimpleNamespace(cuda_stream=next(streams))

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(moe_interleaved, "_load_library", lambda: SimpleNamespace(
        grain_moe_interleaved_launch=lambda *args: calls.append(args) or 0))
    for first in (100, 200):
        x = Tensor(first, (4096, 1536))
        weight = Tensor(first + 1, (32, 2048, 1536))
        others = [Tensor(first + i) for i in range(2, 9)]
        others[5].shape = (32, 576, 1024)
        q, s = moe_interleaved._launch_prevalidated(x, weight, *others,
                    top_k=4, consumer_warps=16, persistent_blocks=20)
        assert q is others[5] and s is others[6]
    for first, stream, args in zip((100, 200), (71, 83), calls):
        expected = [first + i for i in (0, 2, 1, 3, 4, 5, 6, 7, 8)]
        assert args == (*expected, 576, 1024, 1536, 32, 4096, 4, 16, 20, stream)


def test_interleaved_launch_failure_reported_cpu(monkeypatch):
    tensor = SimpleNamespace(shape=(32, 64, 1536), device=object(), data_ptr=lambda: 16)
    weight = SimpleNamespace(shape=(32, 2048, 1536), data_ptr=lambda: 32)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(moe_interleaved, "_load_library", lambda: SimpleNamespace(
        grain_moe_interleaved_launch=lambda *args: -10001,
        grain_moe_interleaved_error_string=lambda code: b"TMA descriptor rejected"))
    with pytest.raises(RuntimeError, match="TMA descriptor rejected"):
        moe_interleaved._launch_prevalidated(tensor, weight, tensor, tensor, tensor,
                    tensor, tensor, tensor, tensor, top_k=4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("intermediate", [256, 768])
@pytest.mark.parametrize("storage", ["ragged", "packed"])
def test_interleaved_swiglu_quantization_retains_all_bits(dtype, intermediate, storage):
    _require_cuda()
    generator = torch.Generator().manual_seed(20261021)
    ids = torch.tensor([[0, 0, 1], [2, 1, 2], [1, 2, 0], [2, 2, 0],
                        [1, 0, 1], [0, 1, 2], [2, 2, 1]], dtype=torch.int64, device="cuda")
    mask = torch.tensor([True, False, True, True, False, True, True], device="cuda")
    ids[~mask] = torch.iinfo(torch.int64).min
    routes = kernels.prepare_routes(ids, 5, mask)
    capacity = 17 if storage == "packed" else None
    shape = (5, capacity, 2 * intermediate) if capacity else (ids.numel(), 2 * intermediate)
    split = torch.randn(shape, generator=generator).to(dtype).cuda()
    split.reshape(-1, 2 * intermediate)[0].zero_()
    # Unwritten ragged tails and packed padding must never feed reductions.
    if capacity:
        for expert, count in enumerate(routes.counts.cpu().tolist()):
            split[expert, count:].fill_(float("nan"))
    else:
        split[int(routes.counts.sum()):].fill_(float("nan"))
    paired = split.index_select(-1, _permutation(intermediate, "cuda")).contiguous()
    expected = kernels.swiglu_quantize(split, routes, packed_capacity=capacity)
    actual = kernels.swiglu_quantize(paired, routes, packed_capacity=capacity,
                                     gate_up_layout="interleaved")
    for observed, wanted in zip(actual, expected):
        assert torch.equal(observed.view(torch.uint8), wanted.view(torch.uint8))


@pytest.mark.parametrize("backend", ["triton", "auto"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("h,i", [(256, 256), (768, 512)])
def test_interleaved_ffn_matches_split_with_mask_duplicates_and_empty_experts(backend, dtype, h, i):
    _require_cuda()
    from test_moe_ffn import make_case, reference

    case = make_case(tokens=9, h=h, i=i, dtype=dtype)
    case["token_mask"] = torch.arange(9, device="cuda") % 3 != 0
    case["topk_ids"][~case["token_mask"]] = torch.iinfo(torch.int64).min
    case["scale_x"][~case["token_mask"]] = float("nan")
    case["topk_weights"][~case["token_mask"]] = float("nan")
    case["scale_x"][1::3].mul_(-1)
    expected = moe_ffn(**case, backend=backend)
    actual = moe_ffn(**_interleave_case(case), backend=backend)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    torch.testing.assert_close(actual.cpu(), reference(case), rtol=0.015, atol=0.025)


@pytest.mark.parametrize("kind", ["empty", "all_masked"])
def test_interleaved_ffn_zero_valid_tokens(kind):
    _require_cuda()
    from test_moe_ffn import make_case

    case = _interleave_case(make_case(tokens=7, h=256))
    if kind == "empty":
        for name in ("hidden_states", "scale_x", "topk_ids", "topk_weights"):
            case[name] = case[name][:0]
    case["token_mask"] = torch.zeros(case["hidden_states"].shape[0], device="cuda", dtype=torch.bool)
    case["topk_ids"].fill_(torch.iinfo(torch.int64).max)
    result = moe_ffn(**case)
    assert result.shape == case["hidden_states"].shape
    assert torch.count_nonzero(result).item() == 0


def test_interleaved_native_dispatch_consumes_passed_weights_without_repacking(monkeypatch):
    _require_interleaved_native()
    from grain_gemm.kernels import moe_interleaved
    from test_moe_ffn import make_case

    case = make_case(tokens=256, h=256, experts=2)
    case["topk_ids"] = (torch.arange(256, device="cuda") % 2)[:, None].contiguous()
    case["topk_weights"] = torch.ones((256, 1), device="cuda")
    paired = _interleave_case(case)
    seen = []
    launch = moe_interleaved._launch_prevalidated

    def tracked(x, weight, scale_x, scale_weight, *args, **kwargs):
        assert x is paired["hidden_states"] and scale_x is paired["scale_x"]
        assert weight.data_ptr() == paired["w_gate_up"].data_ptr()
        assert scale_weight is paired["scale_gate_up"]
        seen.append(True)
        return launch(x, weight, scale_x, scale_weight, *args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Interleaved weights reached the split native first projection")

    monkeypatch.setattr(moe_interleaved, "_launch_prevalidated", tracked)
    monkeypatch.setattr(moe_gather, "_launch_prevalidated", unexpected)
    for backend in ("auto", "cutlass"):
        actual = moe_ffn(**paired, backend=backend)
        expected = moe_ffn(**case, backend="triton")
        torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.025)
    assert seen == [True, True]


@pytest.mark.parametrize("backend", ["auto", "cutlass"])
def test_interleaved_graph_replay_tracks_changing_routes(backend):
    _require_cuda()
    if backend == "cutlass":
        _require_interleaved_native()
    from test_moe_ffn import make_case

    case = make_case(tokens=7, h=256)
    case["token_mask"] = torch.ones(7, device="cuda", dtype=torch.bool)
    paired = _interleave_case(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            moe_ffn(**paired, backend=backend)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = moe_ffn(**paired, backend=backend)
    for expert, enabled in ((4, True), (0, False), (1, True)):
        case["topk_ids"].fill_(expert)
        case["token_mask"].fill_(enabled)
        graph.replay()
        expected = moe_ffn(**case, backend="triton")
        torch.testing.assert_close(result, expected, rtol=0.015, atol=0.025)


@pytest.mark.parametrize("backend", ["triton", "auto", "cutlass"])
def test_interleaved_nondefault_stream(backend):
    _require_cuda()
    if backend == "cutlass":
        _require_interleaved_native()
    from test_moe_ffn import make_case

    case = make_case(tokens=9, h=256)
    paired = _interleave_case(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = moe_ffn(**paired, backend=backend)
    torch.cuda.current_stream().wait_stream(stream)
    expected = moe_ffn(**case, backend="triton")
    torch.testing.assert_close(output, expected, rtol=0.015, atol=0.025)
