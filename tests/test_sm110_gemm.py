"""Thor TMEM pipeline, public scale layout, dispatch isolation and graph tests."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from grain_gemm import int8_gemm, get_kernel_config
from grain_gemm.kernels import cuda
from test_int8_gemm import inputs, reference

TCGEN_IDS = range(17, len(cuda.SM110_CONFIGS))


def require_thor():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Requires NVIDIA Thor SM110")
    if not cuda.is_available():
        pytest.skip("Build the SM110 CUDA backend")


@pytest.fixture(scope="module", params=[256, 512, 768, 4352])
def thor_case(request):
    require_thor()
    ts = list(
        inputs(256, 256, request.param, 256, device="cuda", layout="column_major_b")
    )
    # Every row, column and group differs; exercise signed and zero scales too.
    ts[2][::3] *= -1
    ts[3][:, ::5] = 0
    expected = reference(*ts, 256).bfloat16()
    return ts, expected


@pytest.mark.parametrize("config_id", TCGEN_IDS)
def test_tcgen_configs_preserve_public_scale_layout(thor_case, config_id):
    ts, expected = thor_case
    out = torch.empty(expected.shape, device="cuda", dtype=torch.bfloat16)
    cuda.launch(*ts, out, config_id)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("config_id", TCGEN_IDS)
def test_tcgen_exact_cancellation_and_extrema(config_id):
    require_thor()
    a = torch.full((256, 512), -128, device="cuda", dtype=torch.int8)
    b = torch.full((256, 512), -128, device="cuda", dtype=torch.int8).T
    sa = torch.ones((256, 2), device="cuda")
    sb = torch.ones((2, 256), device="cuda")
    out = torch.empty((256, 256), device="cuda", dtype=torch.bfloat16)
    cuda.launch(a, b, sa, sb, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**23), rtol=0, atol=0)
    # Two contributions cancel except for one INT8 product. FMA must retain 2^-36.
    a.fill_(1)
    b.fill_(1)
    a[:, 0] = 2
    sa.fill_(2**-18)
    sb[0].fill_(2**-18)
    sb[1].fill_(-(2**-18))
    cuda.launch(a, b, sa, sb, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**-36), rtol=0, atol=0)


@pytest.mark.parametrize("config_id", TCGEN_IDS)
def test_tcgen_graph_on_nondefault_stream(config_id):
    require_thor()
    ts = inputs(256, 256, 768, 256, device="cuda", layout="column_major_b")
    out = torch.empty((256, 256), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda.launch(*ts, out, config_id)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cuda.launch(*ts, out, config_id)
    graph.replay()
    torch.testing.assert_close(out.cpu(), reference(*ts, 256).bfloat16())
    ts[0].zero_()
    graph.replay()
    assert torch.count_nonzero(out).item() == 0
    ts[0].fill_(1)
    ts[1].fill_(-2)
    ts[2].fill_(0.125)
    ts[3].fill_(-0.25)
    graph.replay()
    torch.testing.assert_close(out, torch.full_like(out, 48), rtol=0, atol=0)


@pytest.mark.parametrize("config_id", TCGEN_IDS)
def test_tcgen_partial_swizzle_band(config_id):
    require_thor()
    cfg = cuda.SM110_CONFIGS[config_id]
    ts = inputs(
        cfg["block_m"] * 9,
        cfg["block_n"] * 3,
        768,
        256,
        device="cuda",
        layout="column_major_b",
    )
    out = torch.empty(
        (ts[0].shape[0], ts[1].shape[1]), device="cuda", dtype=torch.bfloat16
    )
    cuda.launch(*ts, out, config_id)
    torch.testing.assert_close(out.cpu(), reference(*ts, 256).bfloat16())


def test_thor_dispatch_uses_independent_measurements(monkeypatch):
    import grain_gemm.gemm as impl
    from test_cutlass_gemm import empty_dispatch_inputs

    ts = empty_dispatch_inputs(256, 1024, 1536)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (11, 0)
    )
    monkeypatch.setattr(cuda, "is_available", lambda device=None: True)
    triton = dict(
        backend="triton", block_m=64, block_n=64, num_warps=4, num_stages=2, swizzle=8
    )
    entry = dict(
        selected=dict(backend="cuda", config_id=19),
        cuda=dict(backend="cuda", config_id=19),
        triton=triton,
    )
    monkeypatch.setattr(
        impl,
        "_thor_dispatch_table",
        lambda: dict(square_configs={}, rectangular_configs={"256x1024x1536": entry}),
    )
    monkeypatch.setattr(
        impl, "_dispatch_table", lambda: pytest.fail("Thor read GB10 tuning table")
    )
    a, b, sa, sb = ts
    cfg = get_kernel_config(a, b, scale_a=sa, scale_b=sb)
    assert (
        cfg["backend"] == "cuda"
        and cfg["config_id"] == 19
        and cfg["policy"] == "sm110_g256_measured"
    )
    monkeypatch.setattr(cuda, "is_available", lambda device=None: False)
    cfg = get_kernel_config(a, b, scale_a=sa, scale_b=sb)
    assert cfg["backend"] == "triton" and cfg["block_m"] == 64
    with pytest.raises(RuntimeError):
        get_kernel_config(a, b, scale_a=sa, scale_b=sb, backend="cuda")


@pytest.mark.parametrize("backend", ["auto", "cuda", "triton", "cutlass"])
def test_public_api_keeps_scale_strides_and_inputs(backend):
    require_thor()
    if backend == "cutlass":
        from grain_gemm.kernels import cutlass

        if not cutlass.is_available():
            pytest.skip("Build the SM110 CUTLASS backend")
    ts = inputs(256, 256, 768, 256, device="cuda", layout="column_major_b")
    saved = [t.clone() for t in ts]
    actual = int8_gemm(*ts, backend=backend)
    assert actual.shape == (256, 256) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.cpu(), reference(*ts, 256).bfloat16())
    for t, old in zip(ts, saved):
        torch.testing.assert_close(t, old, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(64, 64, 256), (192, 128, 512)])
def test_tcgen_minimum_and_m64_tiles(shape):
    require_thor()
    ts = inputs(*shape, 256, device="cuda", layout="column_major_b")
    actual = int8_gemm(*ts, backend="cuda")
    torch.testing.assert_close(actual.cpu(), reference(*ts, 256).bfloat16())


def test_tma_accepts_existing_16_byte_alignment_and_scale_offsets():
    require_thor()
    original = inputs(256, 256, 768, 256, device="cuda", layout="column_major_b")
    a, b, sa, sb = original

    def offset_copy(tensor, offset):
        backing = torch.empty(
            tensor.numel() + offset, device="cuda", dtype=tensor.dtype
        )
        result = backing[offset:].view(tensor.shape)
        result.copy_(tensor)
        return result

    # The GB10 native contract requires 16-byte operand alignment, not 128.
    a = offset_copy(a, 16)
    b = offset_copy(b.T, 16).T
    sa, sb = offset_copy(sa, 1), offset_copy(sb, 1)
    assert a.data_ptr() % 128 == b.data_ptr() % 128 == 16
    actual = int8_gemm(a, b, sa, sb, backend="cuda")
    torch.testing.assert_close(actual.cpu(), reference(*original, 256).bfloat16())


def test_thor_source_hash_covers_shared_warp_dependency(monkeypatch, tmp_path):
    from grain_gemm.kernels import _native

    root = tmp_path / "csrc"
    source = root / "sm110/int8_g256_cute.cu"
    dependency = root / "sm12x/int8_g256_cute.cu"
    for path in (source, dependency):
        path.parent.mkdir(parents=True)
        path.write_text("source")
    monkeypatch.setattr(
        _native, "cuda_sources", lambda architecture: (source, dependency)
    )
    before = _native.cuda_source_hash("sm_110")
    dependency.write_text("changed")
    after = _native.cuda_source_hash("sm_110")
    assert before["sm110/int8_g256_cute.cu"] == after["sm110/int8_g256_cute.cu"]
    assert before["sm12x/int8_g256_cute.cu"] != after["sm12x/int8_g256_cute.cu"]
