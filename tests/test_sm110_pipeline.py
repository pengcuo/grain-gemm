"""Independent Thor CuTe pipelines: group math, launch state and API layouts."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import get_kernel_config, int8_gemm
from grain_gemm.kernels import cuda, cutlass
from test_int8_gemm import inputs, reference

PIPELINE_IDS = tuple(
    index
    for index, config in enumerate(cuda.SM110_CONFIGS)
    if config.get("mma") == "tcgen05_ws"
)


def require_pipeline():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Requires NVIDIA Thor SM110")
    if not cuda.is_available():
        pytest.skip("Build the SM110 CuTe backend")


def signed_case(m, n, k):
    tensors = list(inputs(m, n, k, 256, device="cuda", layout="column_major_b"))
    rows = torch.arange(m, device="cuda")[:, None]
    cols = torch.arange(n, device="cuda")[None, :]
    groups = torch.arange(k // 256, device="cuda")
    # Binary fractions keep the reference rounding bounded. Prime periods make
    # row, column and group indexing observable, including zero/negative scales.
    tensors[2].copy_(((5 * rows + 7 * groups) % 31 - 15).float() * 0.03125)
    tensors[3].copy_(((11 * cols + 13 * groups[:, None]) % 37 - 18).float() * 0.0625)
    return tensors


def launch(tensors, config_id):
    out = torch.empty(
        (tensors[0].shape[0], tensors[1].shape[1]),
        device=tensors[0].device,
        dtype=torch.bfloat16,
    )
    cuda.launch(*tensors, out, config_id)
    return out


@pytest.fixture(scope="module", params=[256, 512, 768, 4352])
def grouped_case(request):
    require_pipeline()
    # Complete two-SM tiles exercise the independent pipeline itself, including
    # minimum K and repeated operand/accumulator ring wraps.
    tensors = signed_case(256, 512, request.param)
    return tensors, reference(*tensors, 256).bfloat16()


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
def test_pipeline_group_math_matches_independent_cpu_reference(grouped_case, config_id):
    tensors, expected = grouped_case
    torch.testing.assert_close(launch(tensors, config_id).cpu(), expected)


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
@pytest.mark.parametrize(
    "shape", [(64, 64, 256), (64, 320, 512), (320, 64, 768),
              (576, 832, 512), (256, 128, 256), (256, 128, 768),
              (256, 384, 4352)]
)
def test_pipeline_ids_preserve_native_minimum_and_tail_contract(shape, config_id):
    require_pipeline()
    tensors = signed_case(*shape)
    # A pipeline may delegate these shapes to the scalar-compatible CuTe path;
    # the established native API still accepts positive M/N multiples of 64.
    torch.testing.assert_close(
        launch(tensors, config_id).cpu(), reference(*tensors, 256).bfloat16()
    )


@pytest.mark.parametrize(
    "config_id", [i for i in PIPELINE_IDS if cuda.SM110_CONFIGS[i]["cluster_m"] == 2]
)
def test_pipeline_half_tile_reuses_persistent_cluster_with_odd_group_count(config_id):
    require_pipeline()
    # Eleven M tiles and two N tiles require multiple waves on Thor's ten
    # clusters. Distinct signed scales detect stale rings between output tiles.
    tensors = signed_case(2816, 384, 768)
    torch.testing.assert_close(
        launch(tensors, config_id).cpu(), reference(*tensors, 256).bfloat16()
    )


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
def test_pipeline_extrema_and_fp32_cancellation_before_bf16(config_id):
    require_pipeline()
    a = torch.full((256, 512), -128, device="cuda", dtype=torch.int8)
    b = torch.full((256, 512), -128, device="cuda", dtype=torch.int8).t()
    sa = torch.ones((256, 2), device="cuda")
    sb = torch.ones((2, 256), device="cuda")
    tensors = a, b, sa, sb
    out = launch(tensors, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**23), rtol=0, atol=0)

    # The groups differ by one integer product. Converting each group's scaled
    # term to BF16 early loses this exactly representable FP32 residual.
    a.fill_(1)
    b.fill_(1)
    a[:, 0] = 2
    sa.fill_(2**-18)
    sb[0].fill_(2**-18)
    sb[1].fill_(-(2**-18))
    cuda.launch(*tensors, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, 2**-36), rtol=0, atol=0)

    # Each dot is 257. The first scaled dot rounds to 257 + 2^-15; the
    # second fused multiply-add retains -2^-23. Separate multiply/add loses
    # that residual by rounding both scaled dots to equal magnitudes first.
    a[:, 256] = 2
    sa.fill_(1)
    sb[0].fill_(1 + 2**-23)
    sb[1].fill_(-(1 + 2**-23))
    cuda.launch(*tensors, out, config_id)
    torch.testing.assert_close(out, torch.full_like(out, -(2**-23)), rtol=0, atol=0)


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
@pytest.mark.parametrize("weight_scale_offset", [0, 1])
def test_pipeline_keeps_existing_operand_and_scale_offset_contract(
    config_id, weight_scale_offset
):
    require_pipeline()
    original = signed_case(256, 512, 768)

    def offset_copy(tensor, offset):
        backing = torch.empty(
            tensor.numel() + offset, device="cuda", dtype=tensor.dtype
        )
        result = backing[offset:].view(tensor.shape)
        result.copy_(tensor)
        return result

    a, b, sa, sb = original
    tensors = (
        offset_copy(a, 16),
        offset_copy(b.t(), 16).t(),
        offset_copy(sa, 1),
        offset_copy(sb, weight_scale_offset),
    )
    assert tensors[0].data_ptr() % 128 == tensors[1].data_ptr() % 128 == 16
    assert tensors[2].data_ptr() % 16 == 4
    if weight_scale_offset:
        assert tensors[3].data_ptr() % 16 == 4
    saved = [tensor.clone() for tensor in tensors]
    torch.testing.assert_close(
        launch(tensors, config_id).cpu(), reference(*original, 256).bfloat16()
    )
    for tensor, old in zip(tensors, saved):
        torch.testing.assert_close(tensor, old, rtol=0, atol=0)


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
def test_pipeline_graph_on_nondefault_stream_reads_updated_operands_and_scales(
    config_id,
):
    require_pipeline()
    tensors = signed_case(256, 512, 768)
    out = torch.empty((256, 512), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda.launch(*tensors, out, config_id)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cuda.launch(*tensors, out, config_id)
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


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
def test_pipeline_two_streams_keep_parameters_and_outputs_independent(config_id):
    require_pipeline()
    first = signed_case(512, 512, 1792)
    second = signed_case(512, 512, 1792)
    second[0].fill_(3)
    second[1].fill_(-5)
    second[2].mul_(-0.5)
    second[3].mul_(2)
    cases = first, second
    expected = [reference(*tensors, 256).bfloat16() for tensors in cases]
    outputs = [
        torch.empty((512, 512), device="cuda", dtype=torch.bfloat16) for _ in cases
    ]
    streams = torch.cuda.Stream(), torch.cuda.Stream()
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(4):
        for stream, tensors, out in zip(streams, cases, outputs):
            with torch.cuda.stream(stream):
                cuda.launch(*tensors, out, config_id)
    for stream in streams:
        stream.synchronize()
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), wanted)


@pytest.mark.parametrize("config_id", PIPELINE_IDS)
def test_public_cuda_pipeline_runs_without_cutlass_backend(monkeypatch, config_id):
    require_pipeline()
    import grain_gemm.gemm as implementation

    tensors = signed_case(256, 512, 768)
    selected = dict(backend="cuda", config_id=config_id)
    entry = dict(
        selected=selected,
        cuda=selected,
        triton=dict(
            backend="triton",
            block_m=32,
            block_n=64,
            num_warps=4,
            num_stages=2,
            swizzle=8,
        ),
    )
    monkeypatch.setattr(
        implementation,
        "_thor_dispatch_table",
        lambda: {"square_configs": {}, "rectangular_configs": {"256x512x768": entry}},
    )
    monkeypatch.setattr(cutlass, "is_available", lambda device=None: False)
    monkeypatch.setattr(
        cutlass,
        "launch",
        lambda *args, **kwargs: pytest.fail("Independent CuTe called CUTLASS"),
    )
    config = get_kernel_config(
        *tensors[:2], scale_a=tensors[2], scale_b=tensors[3], backend="cuda"
    )
    assert config["backend"] == "cuda" and config["config_id"] == config_id
    torch.testing.assert_close(
        int8_gemm(*tensors, backend="cuda").cpu(), reference(*tensors, 256).bfloat16()
    )
