"""Thor CUTLASS pipeline: GB10 scales, tile tails, graphs and stream safety."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import get_kernel_config, int8_gemm
from grain_gemm.kernels import cutlass, cuda
from test_cutlass_gemm import empty_dispatch_inputs
from test_int8_gemm import inputs, reference

THOR_CONFIG_IDS = (32, 33)


def require_thor_cutlass():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Requires NVIDIA Thor SM110")
    if not cutlass.is_available():
        pytest.skip("Build the SM110 CUTLASS backend")


def signed_case(m, n, k):
    tensors = list(inputs(m, n, k, 256, device="cuda", layout="column_major_b"))
    # Prime periods and binary fractions make row, column and group offsets
    # observable, including sign changes and rows/columns with zero scales.
    rows = torch.arange(m, device="cuda")[:, None]
    cols = torch.arange(n, device="cuda")[None, :]
    groups = torch.arange(k // 256, device="cuda")
    tensors[2].copy_(((3 * rows + 5 * groups) % 23 - 11).float() * 0.125)
    tensors[3].copy_(((7 * cols + 11 * groups[:, None]) % 29 - 14).float() * 0.0625)
    return tensors


@pytest.fixture(scope="module", params=[256, 512, 768, 4352])
def grouped_case(request):
    require_thor_cutlass()
    tensors = signed_case(192, 320, request.param)
    return tensors, reference(*tensors, 256).bfloat16()


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
def test_cutlass_pipeline_preserves_group_scales_and_tile_tails(
    grouped_case, config_id
):
    tensors, expected = grouped_case
    out = torch.empty(expected.shape, device="cuda", dtype=torch.bfloat16)
    cutlass.launch(*tensors, out, config_id)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
@pytest.mark.parametrize("shape", [(64, 64, 256), (64, 320, 768), (320, 64, 512)])
def test_cutlass_pipeline_supports_minimum_and_partial_tiles(shape, config_id):
    require_thor_cutlass()
    tensors = signed_case(*shape)
    out = torch.empty(shape[:2], device="cuda", dtype=torch.bfloat16)
    cutlass.launch(*tensors, out, config_id)
    torch.testing.assert_close(out.cpu(), reference(*tensors, 256).bfloat16())


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
def test_cutlass_pipeline_exact_extrema_and_cancellation(config_id):
    require_thor_cutlass()
    a = torch.full((128, 512), -128, device="cuda", dtype=torch.int8)
    b = torch.full((128, 512), -128, device="cuda", dtype=torch.int8).t()
    sa = torch.ones((128, 2), device="cuda")
    sb = torch.ones((2, 128), device="cuda")
    out = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(a, b, sa, sb, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**23), rtol=0, atol=0)
    a.fill_(1)
    b.fill_(1)
    a[:, 0] = 2
    sa.fill_(2**-18)
    sb[0].fill_(2**-18)
    sb[1].fill_(-(2**-18))
    cutlass.launch(a, b, sa, sb, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**-36), rtol=0, atol=0)


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
def test_cutlass_pipeline_nondefault_stream_graph_reads_updated_inputs(config_id):
    require_thor_cutlass()
    tensors = signed_case(192, 320, 768)
    out = torch.empty((192, 320), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cutlass.launch(*tensors, out, config_id)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cutlass.launch(*tensors, out, config_id)
    graph.replay()
    torch.testing.assert_close(out.cpu(), reference(*tensors, 256).bfloat16())
    tensors[0].zero_()
    graph.replay()
    assert torch.count_nonzero(out).item() == 0
    tensors[0].fill_(1)
    tensors[1].fill_(-2)
    tensors[2].fill_(0.125)
    tensors[3].fill_(-0.25)
    graph.replay()
    torch.testing.assert_close(out, torch.full_like(out, 48), rtol=0, atol=0)


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
def test_cutlass_concurrent_streams_keep_launch_state_and_output_independent(config_id):
    require_thor_cutlass()
    first = signed_case(512, 768, 4352)
    second = signed_case(512, 768, 4352)
    second[0].fill_(3)
    second[1].fill_(-5)
    second[2].mul_(-0.5)
    second[3].mul_(2)
    cases = [first, second]
    expected = [reference(*tensors, 256).bfloat16() for tensors in cases]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [
        torch.empty((512, 768), device="cuda", dtype=torch.bfloat16) for _ in cases
    ]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    # Outstanding launches must keep their descriptors and kernel state separate
    # while the two streams consume different operands and scales.
    for _ in range(4):
        for stream, tensors, out in zip(streams, cases, outputs):
            with torch.cuda.stream(stream):
                cutlass.launch(*tensors, out, config_id)
    for stream in streams:
        stream.synchronize()
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), wanted)


@pytest.mark.parametrize("config_id", THOR_CONFIG_IDS)
@pytest.mark.parametrize("scale_offset", [0, 1])
def test_cutlass_keeps_operand_and_scale_storage_offset_contract(
    config_id, scale_offset
):
    require_thor_cutlass()
    original = signed_case(192, 320, 768)

    def offset_copy(tensor, offset):
        backing = torch.empty(
            tensor.numel() + offset, device="cuda", dtype=tensor.dtype
        )
        result = backing[offset:].view(tensor.shape)
        result.copy_(tensor)
        return result

    a, b, sa, sb = original
    a = offset_copy(a, 16)
    b = offset_copy(b.t(), 16).t()
    sa = offset_copy(sa, 1)
    sb = offset_copy(sb, scale_offset)
    assert a.data_ptr() % 128 == b.data_ptr() % 128 == 16
    assert sa.data_ptr() % 16 == 4
    if scale_offset:
        assert sb.data_ptr() % 16 == 4
    out = torch.empty((192, 320), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(a, b, sa, sb, out, config_id)
    torch.testing.assert_close(out.cpu(), reference(*original, 256).bfloat16())
    saved = [t.clone() for t in (a, b, sa, sb)]
    actual = int8_gemm(a, b, sa, sb, backend="cutlass")
    torch.testing.assert_close(actual.cpu(), reference(*original, 256).bfloat16())
    for tensor, old in zip((a, b, sa, sb), saved):
        torch.testing.assert_close(tensor, old, rtol=0, atol=0)


@pytest.fixture
def simulated_thor(monkeypatch):
    import grain_gemm.gemm as implementation

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (11, 0)
    )
    monkeypatch.setattr(cutlass, "is_available", lambda device=None: True)
    monkeypatch.setattr(cuda, "is_available", lambda device=None: True)
    monkeypatch.setattr(
        implementation, "_dispatch_table", lambda: pytest.fail("Thor read GB10 table")
    )
    monkeypatch.setattr(
        implementation,
        "_cutlass_dispatch_table",
        lambda: pytest.fail("Thor read GB10 CUTLASS table"),
    )
    triton = dict(
        backend="triton", block_m=64, block_n=128, num_warps=4, num_stages=2, swizzle=8
    )
    entry = dict(
        selected=dict(backend="cutlass", config_id=32),
        cutlass=dict(backend="cutlass", config_id=32),
        cuda=dict(backend="cuda", config_id=19),
        triton=triton,
    )
    table = dict(square_configs={}, rectangular_configs={"192x320x768": entry})
    monkeypatch.setattr(implementation, "_thor_dispatch_table", lambda: table)
    return entry


def select(tensors, backend="auto"):
    return get_kernel_config(
        *tensors[:2], scale_a=tensors[2], scale_b=tensors[3], backend=backend
    )


def test_thor_auto_uses_measured_cutlass_and_available_fallbacks(
    simulated_thor, monkeypatch
):
    tensors = empty_dispatch_inputs(192, 320, 768)
    config = select(tensors)
    assert config["backend"] == "cutlass" and config["config_id"] == 32
    assert config["policy"] == "sm110_g256_measured"
    monkeypatch.setattr(cutlass, "is_available", lambda device=None: False)
    config = select(tensors)
    assert config["backend"] == "cuda" and config["config_id"] == 19
    monkeypatch.setattr(cuda, "is_available", lambda device=None: False)
    config = select(tensors)
    assert config["backend"] == "triton" and config["block_n"] == 128
    with pytest.raises(RuntimeError, match="(?i)(not built|unavailable)"):
        select(tensors, backend="cutlass")


def test_thor_auto_preserves_measured_triton_fallback(simulated_thor, monkeypatch):
    simulated_thor["fallback_selected"] = simulated_thor["triton"]
    monkeypatch.setattr(cutlass, "is_available", lambda device=None: False)
    config = select(empty_dispatch_inputs(192, 320, 768))
    assert config["backend"] == "triton" and config["block_n"] == 128


def test_unmeasured_thor_auto_stays_portable(simulated_thor):
    tensors = empty_dispatch_inputs(128, 128, 768)
    assert select(tensors)["backend"] == "triton"
    explicit = select(tensors, backend="cutlass")
    assert explicit["backend"] == "cutlass" and explicit["config_id"] == 32


@pytest.mark.parametrize("backend", ["auto", "cutlass"])
def test_four_byte_weight_scale_offset_keeps_public_native_api(simulated_thor, backend):
    tensors = list(empty_dispatch_inputs(192, 320, 768))
    backing = torch.empty(tensors[3].numel() + 1, dtype=torch.float32)
    tensors[3] = backing[1:].view(tensors[3].shape)
    assert tensors[3].is_contiguous() and tensors[3].data_ptr() % 16 == 4
    config = select(tensors, backend=backend)
    if backend == "cutlass":
        assert config["backend"] == "cutlass" and config["config_id"] == 0
    else:
        assert config["backend"] == "cuda" and config["config_id"] == 19


@pytest.mark.parametrize(
    "capability,expected_length", [((11, 0), 34), ((12, 0), 32), ((12, 1), 32)]
)
def test_cutlass_target_configs_preserve_gb10_ids(
    monkeypatch, capability, expected_length
):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    configs = cutlass.configs_for_device(torch.device("cuda:2"))
    assert len(cutlass.CONFIGS) == 32
    assert len(configs) == expected_length
    assert configs[:32] == cutlass.CONFIGS


@pytest.mark.parametrize(
    "dependency",
    [
        "sm110/include/cutlass/arch/reg_reconfig.h",
        "sm12x/cutlass/collective.hpp",
    ],
)
def test_thor_cutlass_hash_tracks_shadow_and_legacy_dependencies(
    monkeypatch, tmp_path, dependency
):
    from grain_gemm.kernels import _native

    root = tmp_path / "csrc"
    source = root / "sm110/int8_g256_cutlass.cu"
    shared = root / dependency
    for path in (source, shared):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    monkeypatch.setattr(_native, "__file__", str(tmp_path / "_native.py"))
    monkeypatch.setattr(
        _native, "cutlass_sources", lambda architecture: (source, shared)
    )
    before = _native.cutlass_source_hash("sm_110")
    shared.write_text("changed")
    after = _native.cutlass_source_hash("sm_110")
    assert before["sm110/int8_g256_cutlass.cu"] == after["sm110/int8_g256_cutlass.cu"]
    assert before[dependency] != after[dependency]
