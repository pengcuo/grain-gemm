"""Regression checks for the optional high-level CUTLASS G256 backend."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import get_kernel_config, int8_gemm
from grain_gemm.kernels.cutlass import CONFIGS as CUTLASS_CONFIGS
from test_int8_gemm import cuda, inputs, reference


ALL_CONFIGS = range(len(CUTLASS_CONFIGS))


def require_cutlass():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("The optional CUTLASS backend requires SM121")
    from grain_gemm.kernels import cutlass

    if not cutlass.is_available():
        pytest.skip("The optional CUTLASS library has not been built")
    return cutlass


def select_cutlass(tensors, *, group_size=256, output_dtype=torch.bfloat16,
                   backend="cutlass"):
    return get_kernel_config(
        *tensors[:2], scale_a=tensors[2], scale_b=tensors[3],
        group_size=group_size, output_dtype=output_dtype, backend=backend,
    )


def empty_dispatch_inputs(m, n, k, group_size=256):
    """Allocate layout metadata without initializing large dispatch-only inputs."""
    return (
        torch.empty((m, k), dtype=torch.int8),
        torch.empty((n, k), dtype=torch.int8).t(),
        torch.empty((m, k // group_size), dtype=torch.float32),
        torch.empty((k // group_size, n), dtype=torch.float32),
    )


@pytest.fixture
def simulated_cutlass(monkeypatch):
    """Exercise policy checks on real CPU strides without launching kernels."""
    from grain_gemm.kernels import cutlass

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 1))
    monkeypatch.setattr(cutlass, "is_available", lambda: True)
    return cutlass


def test_explicit_cutlass_configuration(simulated_cutlass):
    tensors = inputs(128, 128, 512, 256, layout="column_major_b")
    config = select_cutlass(tensors)
    assert config["backend"] == "cutlass"
    assert config["config_id"] in range(len(simulated_cutlass.CONFIGS))


@pytest.mark.parametrize("shape,cutlass_id,cute_id", [
    ((256, 1024, 1536), 3, 3),
    ((256, 1536, 1024), 5, 5),
    ((256, 2048, 1536), 7, 7),
    ((1024, 1024, 1024), 4, 2),
    ((2048, 2048, 2048), 6, 7),
    ((4096, 4096, 4096), 7, 7),
])
def test_cutlass_uses_its_own_measured_configuration(simulated_cutlass, monkeypatch,
                                                    shape, cutlass_id, cute_id):
    import grain_gemm.gemm as implementation
    from grain_gemm.kernels import cuda as cute

    monkeypatch.setattr(cute, "is_available", lambda: True)
    # Missing the independent optimization table must preserve the original
    # backend-specific measurements in the shared dispatch table.
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: {"configs": {}})
    tensors = empty_dispatch_inputs(*shape)
    config = select_cutlass(tensors)
    assert config["backend"] == "cutlass"
    assert config["config_id"] == cutlass_id
    assert config["policy"] == "sm121_g256_measured"
    # The 1024/2048 squares intentionally choose different CUTLASS and CuTe
    # configurations, catching accidental reuse of entry['cuda']/'selected'.
    for backend in ("cuda", "auto"):
        config = select_cutlass(tensors, backend=backend)
        assert config["backend"] == "cuda"
        assert config["config_id"] == cute_id


@pytest.mark.parametrize("shape,config_id", [((256, 1024, 1536), 15),
                                            ((256, 640, 768), 23)])
def test_separate_cutlass_table_takes_precedence_without_changing_other_backends(
        simulated_cutlass, monkeypatch, shape, config_id):
    import grain_gemm.gemm as implementation
    from grain_gemm.kernels import cuda as cute

    monkeypatch.setattr(cute, "is_available", lambda: True)
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: {"configs": {}})
    tensors = empty_dispatch_inputs(*shape)
    previous = {backend: select_cutlass(tensors, backend=backend)
                for backend in ("auto", "cuda", "triton")}
    key = "x".join(map(str, shape))
    table = {"configs": {key: {
        "config": {"backend": "cutlass", "config_id": config_id},
        "measurement_sha256": "a" * 64,
    }}}
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: table)
    config = select_cutlass(tensors)
    assert config["backend"] == "cutlass"
    assert config["config_id"] == config_id
    for backend, expected in previous.items():
        assert select_cutlass(tensors, backend=backend) == expected


@pytest.mark.parametrize("shape", [(128, 640, 768), (256, 768, 768),
                                   (256, 640, 1024)])
def test_separate_cutlass_table_requires_exact_m_n_k(simulated_cutlass, monkeypatch, shape):
    import grain_gemm.gemm as implementation

    table = {"configs": {"256x640x768": {
        "config": {"backend": "cutlass", "config_id": 23},
        "measurement_sha256": "b" * 64,
    }}}
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: table)
    config = select_cutlass(empty_dispatch_inputs(*shape))
    assert config["backend"] == "cutlass"
    assert config["config_id"] == 7
    assert config["policy"] == "sm121_g256_cutlass_explicit"


@pytest.mark.parametrize("variation", ["group_size", "output_dtype", "layout", "not_built"])
def test_separate_cutlass_table_preserves_backend_requirements(simulated_cutlass,
                                                              monkeypatch, variation):
    import grain_gemm.gemm as implementation

    table = {"configs": {"256x640x768": {
        "config": {"backend": "cutlass", "config_id": 23},
        "measurement_sha256": "c" * 64,
    }}}
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: table)
    group_size = 128 if variation == "group_size" else 256
    output_dtype = torch.float32 if variation == "output_dtype" else torch.bfloat16
    tensors = list(empty_dispatch_inputs(256, 640, 768, group_size))
    if variation == "layout":
        tensors[1] = tensors[1].contiguous()
    if variation == "not_built":
        monkeypatch.setattr(simulated_cutlass, "is_available", lambda: False)
    exception = RuntimeError if variation == "not_built" else ValueError
    with pytest.raises(exception):
        select_cutlass(tensors, group_size=group_size, output_dtype=output_dtype)


def test_cutlass_optimization_table_has_valid_exact_shapes(simulated_cutlass):
    import grain_gemm.gemm as implementation

    configs = implementation._cutlass_dispatch_table()["configs"]
    assert len(configs) == 18
    for key, entry in configs.items():
        dimensions = key.split("x")
        assert len(dimensions) == 3
        m, n, k = map(int, dimensions)
        assert key == f"{m}x{n}x{k}"
        assert min(m, n, k) > 0
        config = entry["config"]
        assert config["backend"] == "cutlass"
        config_id = config["config_id"]
        assert type(config_id) is int and config_id in ALL_CONFIGS
        kernel = CUTLASS_CONFIGS[config_id]
        assert m % kernel["block_m"] == n % kernel["block_n"] == k % 256 == 0
        measurement_hash = entry["measurement_sha256"]
        assert len(measurement_hash) == 64
        assert all(character in "0123456789abcdef" for character in measurement_hash)
        selected = select_cutlass(empty_dispatch_inputs(m, n, k))
        assert selected["backend"] == "cutlass"
        assert selected["config_id"] == config_id


def test_cutlass_missing_build_does_not_silently_fallback(simulated_cutlass, monkeypatch):
    monkeypatch.setattr(simulated_cutlass, "is_available", lambda: False)
    tensors = inputs(128, 128, 512, 256, layout="column_major_b")
    with pytest.raises(RuntimeError, match="(?i)(not built|unavailable)"):
        select_cutlass(tensors)


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (11, 0), (12, 0)])
def test_cutlass_rejects_other_architectures(simulated_cutlass, monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    tensors = inputs(128, 128, 512, 256, layout="column_major_b")
    with pytest.raises(ValueError, match="SM121"):
        select_cutlass(tensors)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_cutlass_rejects_other_group_sizes(simulated_cutlass, group_size):
    tensors = inputs(128, 128, 512, group_size, layout="column_major_b")
    with pytest.raises(ValueError, match="G256"):
        select_cutlass(tensors, group_size=group_size)


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
def test_cutlass_rejects_other_output_dtypes(simulated_cutlass, output_dtype):
    tensors = inputs(128, 128, 512, 256, layout="column_major_b")
    with pytest.raises(ValueError, match="BF16"):
        select_cutlass(tensors, output_dtype=output_dtype)


@pytest.mark.parametrize("shape", [(127, 128, 512), (128, 127, 512),
                                   (128, 128, 511), (0, 128, 512),
                                   (128, 0, 512), (128, 128, 0)])
def test_cutlass_configuration_rejects_unsupported_shapes(simulated_cutlass, shape):
    # The public API may return empty outputs directly; describing a native
    # launch itself still requires nonzero, supported dimensions.
    tensors = inputs(*shape, 256, layout="column_major_b")
    with pytest.raises(ValueError):
        select_cutlass(tensors)


@pytest.mark.parametrize("variation", ["strided_a", "row_major_b", "strided_scale_a",
                                       "strided_scale_b", "unaligned_a", "unaligned_b",
                                       "missing_scales"])
def test_cutlass_rejects_unsupported_layouts(simulated_cutlass, variation):
    tensors = list(inputs(128, 128, 512, 256, layout="column_major_b"))
    if variation.startswith("strided_"):
        index = {"strided_a": 0, "strided_scale_a": 2, "strided_scale_b": 3}[variation]
        source = tensors[index]
        tensors[index] = torch.empty(
            (source.shape[0], 2 * source.shape[1]), dtype=source.dtype
        )[:, ::2]
    elif variation == "row_major_b":
        tensors[1] = tensors[1].contiguous()
    elif variation.startswith("unaligned_"):
        index = 0 if variation == "unaligned_a" else 1
        source = tensors[index]
        backing = torch.empty(source.numel() + 1, dtype=source.dtype)[1:]
        tensors[index] = (backing.view(source.shape) if index == 0
                          else backing.view(source.shape[1], source.shape[0]).t())
        assert tensors[index].data_ptr() % 16 == 1
    else:
        tensors[2:] = [None, None]
    with pytest.raises(ValueError):
        select_cutlass(tensors)


@pytest.mark.parametrize("shape,expected_id", [((256, 1024, 1536), 3),
                                               ((1024, 1024, 1024), 2),
                                               ((128, 128, 512), None)])
@pytest.mark.parametrize("cute_available", [False, True])
def test_cutlass_availability_does_not_change_auto_dispatch(simulated_cutlass,
                                                           monkeypatch, shape,
                                                           expected_id, cute_available):
    from grain_gemm.kernels import cuda as cute

    monkeypatch.setattr(cute, "is_available", lambda: cute_available)
    tensors = inputs(*shape, 256, layout="column_major_b")
    config = select_cutlass(tensors, backend="auto")
    if cute_available and expected_id is not None:
        assert config["backend"] == "cuda"
        assert config["config_id"] == expected_id
    else:
        assert config["backend"] == "triton"


@pytest.fixture(scope="module", params=[256, 512, 1536, 4352])
def cutlass_case(request):
    require_cutlass()
    tensors = inputs(128, 128, request.param, 256,
                     device="cuda", layout="column_major_b")
    expected = reference(*tensors, 256).to(torch.bfloat16)
    return tensors, expected


@cuda
@pytest.mark.parametrize("config_id", ALL_CONFIGS)
def test_cutlass_all_configurations(cutlass_case, config_id):
    cutlass = require_cutlass()
    tensors, expected = cutlass_case
    actual = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(*tensors, actual, config_id=config_id)
    torch.testing.assert_close(actual.cpu(), expected)


@cuda
def test_cutlass_public_api_uses_selected_backend(cutlass_case, monkeypatch):
    cutlass = require_cutlass()
    tensors, expected = cutlass_case
    selected = select_cutlass(tensors)
    original_launch = cutlass.launch
    calls = []

    def record_launch(a, b, scale_a, scale_b, output, config_id):
        calls.append(config_id)
        return original_launch(a, b, scale_a, scale_b, output, config_id)

    monkeypatch.setattr(cutlass, "launch", record_launch)
    actual = int8_gemm(*tensors, group_size=256, output_dtype=torch.bfloat16,
                       backend="cutlass")
    assert calls == [selected["config_id"]]
    assert actual.device == tensors[0].device
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.cpu(), expected)


@cuda
@pytest.mark.parametrize("config_id", ALL_CONFIGS)
@pytest.mark.parametrize("k", [256, 4352])
def test_cutlass_partial_cta_bands(config_id, k):
    cutlass = require_cutlass()
    # Complete compute tiles, but only three M tiles in a grouped grid stripe.
    config = CUTLASS_CONFIGS[config_id]
    m, n = 3 * config["block_m"], 2 * config["block_n"]
    tensors = inputs(m, n, k, 256, device="cuda", layout="column_major_b")
    actual = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(*tensors, actual, config_id=config_id)
    assert_sampled_cutlass_reference(actual, tensors)


def sampled_coordinates(m, n):
    # Probe both sides of 64/128-element tile boundaries and the final CTA.
    rows = sorted({index for index in (0, 63, 64, 127, 128, 255, 256, m - 1)
                   if index < m})
    cols = sorted({index for index in (0, 63, 64, 127, 128, 255, 256, n - 1)
                   if index < n})
    return rows, cols


def assert_sampled_cutlass_reference(actual, tensors):
    a, b, scale_a, scale_b = tensors
    rows, cols = sampled_coordinates(a.shape[0], b.shape[1])
    expected = reference(a[rows], b[:, cols], scale_a[rows], scale_b[:, cols], 256)
    torch.testing.assert_close(actual[rows][:, cols].cpu(), expected.to(actual.dtype))


@pytest.fixture(scope="module", params=[(384, 256, 256), (256, 384, 768),
                                        (384, 256, 1280)])
def multicta_scale_case(request):
    require_cutlass()
    m, n, k = request.param
    tensors = inputs(m, n, k, 256, device="cuda", layout="column_major_b")
    # Values differ across row/column CTA offsets and K-groups. Periodic scales
    # repeating every 64 rows/columns would miss a missing CTA offset.
    a, b, scale_a, scale_b = tensors
    scale_a[1::3].neg_()
    scale_b[:, 2::3].neg_()
    scale_a[2::7] = 0
    scale_b[:, 3::11] = 0
    rows, cols = sampled_coordinates(m, n)
    expected = reference(a[rows], b[:, cols], scale_a[rows], scale_b[:, cols], 256)
    return tensors, rows, cols, expected.to(torch.bfloat16)


@cuda
@pytest.mark.parametrize("config_id", ALL_CONFIGS)
def test_cutlass_multicta_scale_offsets_and_odd_group_pipeline(multicta_scale_case,
                                                             config_id):
    cutlass = require_cutlass()
    tensors, rows, cols, expected = multicta_scale_case
    actual = torch.empty((tensors[0].shape[0], tensors[1].shape[1]),
                         device="cuda", dtype=torch.bfloat16)
    cutlass.launch(*tensors, actual, config_id=config_id)
    torch.testing.assert_close(actual[rows][:, cols].cpu(), expected)


@cuda
@pytest.mark.parametrize("config_id", ALL_CONFIGS)
def test_cutlass_signed_zero_scales_and_int8_extrema(config_id):
    cutlass = require_cutlass()
    rows, cols = torch.arange(128)[:, None], torch.arange(128)[:, None]
    groups = torch.arange(3)[None, :]
    endpoints = torch.tensor([-128, 127], dtype=torch.int8)
    a = endpoints[(rows + groups) % 2].repeat_interleave(256, dim=1).cuda()
    b = endpoints[(cols + groups // 2) % 2].repeat_interleave(256, dim=1).cuda().t()
    scale_a = torch.tensor([0, 0.125, -0.5, 1])[(rows + 2 * groups) % 4].cuda()
    scale_b = torch.tensor([-0.25, 0, 0.5, 2])[(cols + 3 * groups) % 4]
    scale_b = scale_b.t().contiguous().cuda()
    actual = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(a, b, scale_a, scale_b, actual, config_id=config_id)
    expected = reference(a, b, scale_a, scale_b, 256).to(actual.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("config_id", ALL_CONFIGS)
def test_cutlass_fused_scaling_preserves_small_cancellation(config_id):
    cutlass = require_cutlass()
    # First group contributes -1. The second INT32 dot is 4097. Multiplying by
    # float32(1/4097) and adding -1 with FMA leaves exactly 2^-36, rather than 0.
    a = torch.zeros((128, 512), dtype=torch.int8, device="cuda")
    weights = torch.zeros_like(a)
    a[:, 0], weights[:, 0] = -1, 1
    a[:, 256], weights[:, 256] = 127, 32
    a[:, 257], weights[:, 257] = 1, 33
    scale_a = torch.ones((128, 2), device="cuda")
    scale_b = torch.ones((2, 128), device="cuda")
    scale_b[1] = 1 / 4097
    actual = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    cutlass.launch(a, weights.t(), scale_a, scale_b, actual, config_id=config_id)
    torch.testing.assert_close(actual, torch.full_like(actual, 2**-36), rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("config_id", [None, *ALL_CONFIGS])
def test_cutlass_nondefault_stream_observes_prior_writes(config_id):
    cutlass = require_cutlass()
    tensors = inputs(128, 128, 768, 256, device="cuda", layout="column_major_b")
    output = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)

    def launch():
        if config_id is None:
            return int8_gemm(*tensors, output_dtype=torch.bfloat16, backend="cutlass")
        return cutlass.launch(*tensors, output, config_id=config_id)

    launch()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    complete = torch.cuda.Event()
    with torch.cuda.stream(stream):
        a, b, scale_a, scale_b = tensors
        a.fill_(-7)
        b.fill_(5)
        scale_a.fill_(0.125)
        scale_b.fill_(0.25)
        actual = launch()
        complete.record()
    torch.cuda.current_stream().wait_event(complete)
    torch.testing.assert_close(actual.cpu(), torch.full((128, 128), -840.0,
                                                       dtype=torch.bfloat16),
                               rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("config_id", [None, *ALL_CONFIGS])
def test_cutlass_cuda_graph_replay_reads_updated_inputs(config_id):
    cutlass = require_cutlass()
    tensors = inputs(128, 128, 768, 256, device="cuda", layout="column_major_b")
    output = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)

    def launch():
        if config_id is None:
            return int8_gemm(*tensors, output_dtype=torch.bfloat16, backend="cutlass")
        return cutlass.launch(*tensors, output, config_id=config_id)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = launch()
    graph.replay()
    assert_sampled_cutlass_reference(actual, tensors)

    a, b, scale_a, scale_b = tensors
    a.fill_(1)
    b.fill_(-2)
    scale_a.fill_(0.125)
    scale_b.fill_(-0.25)
    graph.replay()
    torch.testing.assert_close(actual.cpu(), torch.full((128, 128), 48.0,
                                                       dtype=torch.bfloat16),
                               rtol=0, atol=0)


@cuda
def test_cutlass_public_api_rejects_non_fp32_scales():
    require_cutlass()
    tensors = list(inputs(128, 128, 512, 256, device="cuda", layout="column_major_b"))
    tensors[2] = tensors[2].half()
    with pytest.raises(TypeError, match="scale_a"):
        int8_gemm(*tensors, output_dtype=torch.bfloat16, backend="cutlass")
