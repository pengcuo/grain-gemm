"""Check GrainGEMM against independently accumulated CPU INT32 group dots."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import get_kernel_config, int8_gemm


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="An SM80-or-newer CUDA device is required",
)


def reference(a, b, scale_a, scale_b, group_size):
    """Keep the inner products exact; apply dequantization only after each dot."""
    a = a.cpu().to(torch.int32)
    b = b.cpu().to(torch.int32)
    scale_a, scale_b = scale_a.cpu(), scale_b.cpu()
    result = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], group_size)):
        partial = a[:, start : start + group_size] @ b[start : start + group_size]
        scales = scale_a[:, group, None] * scale_b[group, None, :]
        result += partial.to(torch.float32) * scales
    return result


def inputs(m, n, k, group_size, *, device="cpu", layout="contiguous"):
    generator = torch.Generator(device=device).manual_seed(2026)
    groups = (k + group_size - 1) // group_size

    def integers(shape):
        return torch.randint(
            -128, 128, shape, dtype=torch.int8, device=device, generator=generator
        )

    def scales(shape):
        return 0.001 + torch.rand(shape, device=device, generator=generator) * 0.05

    if layout == "transposed":
        return (
            integers((k, m)).t(),
            integers((n, k)).t(),
            scales((groups, m)).t(),
            scales((n, groups)).t(),
        )
    if layout == "sliced":
        return tuple(
            make((2 * rows + 1, 2 * cols + 1))[1::2, 1::2]
            for make, rows, cols in (
                (integers, m, k),
                (integers, k, n),
                (scales, m, groups),
                (scales, groups, n),
            )
        )
    a = integers((m, k))
    b = integers((n, k)).t() if layout == "column_major_b" else integers((k, n))
    return a, b, scales((m, groups)), scales((groups, n))


def assert_matches_reference(actual, tensors, group_size):
    expected = reference(*tensors, group_size).to(actual.dtype)
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-3)
    else:
        torch.testing.assert_close(actual.cpu(), expected)


def require_native_cuda():
    """Keep the optional native tests usable without building the CUDA library."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("The optional native CUDA kernels require SM121")
    from grain_gemm.kernels import cuda as native_cuda

    if not native_cuda.is_available():
        pytest.skip("The optional native CUDA library has not been built")
    return native_cuda


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_group_sizes_with_m_n_k_tails(group_size, backend):
    tensors = inputs(19, 37, 2 * group_size + 13, group_size, device="cuda")
    actual = int8_gemm(*tensors, group_size=group_size, backend=backend)
    assert actual.dtype == torch.float32
    assert actual.shape == (19, 37)
    assert actual.device == tensors[0].device
    assert_matches_reference(actual, tensors, group_size)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_aligned_g256_column_major_weights(backend):
    tensors = inputs(256, 128, 768, 256, device="cuda", layout="column_major_b")
    assert tensors[0].is_contiguous()
    assert tensors[1].t().is_contiguous()
    config = get_kernel_config(tensors[0], tensors[1], group_size=256)
    assert isinstance(config, dict)
    assert config
    actual = int8_gemm(*tensors, group_size=256, backend=backend)
    assert_matches_reference(actual, tensors, 256)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("layout", ["transposed", "sliced"])
def test_positive_stride_input_and_scale_views(layout, backend):
    tensors = inputs(33, 65, 269, 128, device="cuda", layout=layout)
    assert all(not tensor.is_contiguous() for tensor in tensors)
    assert all(all(stride > 0 for stride in tensor.stride()) for tensor in tensors)
    if layout == "sliced":
        assert all(tensor.storage_offset() > 0 for tensor in tensors)
    actual = int8_gemm(*tensors, group_size=128, backend=backend)
    assert_matches_reference(actual, tensors, 128)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_distinct_signed_and_zero_row_column_group_scales(backend):
    group_size = 32
    a = torch.ones((3, 69), dtype=torch.int8, device="cuda")
    b = torch.cat(
        [
            torch.full((length, 4), value, dtype=torch.int8)
            for length, value in [(32, 1), (32, -2), (5, 3)]
        ]
    ).cuda()
    scale_a = torch.tensor([[1, -8, 0], [2, 0, 1], [-0.5, 2, 4]], device="cuda")
    scale_b = torch.tensor(
        [[0.5, -2, 0, 1], [-4, 1, 0.25, 2], [2, 0.5, -1, 0]], device="cuda"
    )
    actual = int8_gemm(a, b, scale_a, scale_b, group_size=group_size, backend=backend)
    torch.testing.assert_close(
        actual.cpu(), reference(a, b, scale_a, scale_b, group_size), rtol=0, atol=0
    )


@cuda
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_int8_extrema_accumulate_without_overflow(group_size):
    k = group_size + 7
    a = torch.tensor([-128, 127], dtype=torch.int8, device="cuda")[:, None]
    a = a.expand(2, k).contiguous()
    b = torch.tensor([-128, 127], dtype=torch.int8, device="cuda")[None, :]
    b = b.expand(k, 2).contiguous()
    scale_a = torch.ones((2, 2), device="cuda")
    scale_b = torch.ones((2, 2), device="cuda")
    actual = int8_gemm(a, b, scale_a, scale_b, group_size=group_size)
    torch.testing.assert_close(
        actual.cpu(), reference(a, b, scale_a, scale_b, group_size), rtol=0, atol=0
    )


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_output_dtypes(output_dtype, backend):
    tensors = inputs(17, 29, 103, 64, device="cuda")
    actual = int8_gemm(
        *tensors, group_size=64, output_dtype=output_dtype, backend=backend
    )
    assert actual.dtype == output_dtype
    assert_matches_reference(actual, tensors, 64)


@cuda
@pytest.mark.parametrize("shape", [(0, 7, 65), (7, 0, 65), (7, 9, 0)])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_dimensions(shape, output_dtype):
    m, n, k = shape
    actual = int8_gemm(
        *inputs(m, n, k, 32, device="cuda"),
        group_size=32,
        output_dtype=output_dtype,
    )
    assert actual.shape == (m, n)
    assert actual.dtype == output_dtype
    assert actual.device.type == "cuda"
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("group_size", [16, 0, True, 256.0])
def test_invalid_group_size(group_size):
    with pytest.raises(ValueError):
        int8_gemm(*inputs(2, 3, 32, 256), group_size=group_size)


def test_invalid_backend():
    with pytest.raises(ValueError):
        int8_gemm(*inputs(2, 3, 32, 256), backend="unknown")


@pytest.mark.parametrize(
    "index,replacement,exception",
    [
        (0, lambda t: None, TypeError),
        (0, lambda t: t.float(), TypeError),
        (1, lambda t: t.to(torch.int16), TypeError),
        (2, lambda t: t.half(), TypeError),
        (3, lambda t: t.to(torch.float64), TypeError),
        (0, lambda t: t.unsqueeze(0), ValueError),
        (1, lambda t: t[:-1], ValueError),
        (2, lambda t: t[:, :0], ValueError),
        (3, lambda t: t[:, :-1], ValueError),
    ],
)
def test_invalid_tensor_arguments(index, replacement, exception):
    tensors = list(inputs(2, 3, 32, 256))
    tensors[index] = replacement(tensors[index])
    with pytest.raises(exception):
        int8_gemm(*tensors)


def test_invalid_output_dtype():
    with pytest.raises(TypeError):
        int8_gemm(*inputs(2, 3, 32, 256), output_dtype=torch.int32)


def test_cpu_inputs_are_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        int8_gemm(*inputs(2, 3, 32, 256))


@cuda
@pytest.mark.parametrize("index", range(4))
def test_broadcast_zero_stride_inputs_are_rejected(index):
    tensors = list(inputs(3, 5, 512, 256, device="cuda"))
    tensors[index] = tensors[index][:1, :].expand_as(tensors[index])
    assert tensors[index].stride(0) == 0
    with pytest.raises(ValueError, match="stride"):
        int8_gemm(*tensors)


@cuda
def test_mixed_devices_are_rejected():
    tensors = list(inputs(3, 5, 32, 256, device="cuda"))
    tensors[3] = tensors[3].cpu()
    with pytest.raises(ValueError):
        int8_gemm(*tensors)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton", "cuda"])
def test_cuda_graph_replay_reads_updated_inputs(backend):
    if backend == "cuda":
        require_native_cuda()
    output_dtype = torch.bfloat16 if backend == "cuda" else torch.float32
    tensors = inputs(256, 128, 768, 256, device="cuda", layout="column_major_b")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            int8_gemm(
                *tensors, group_size=256, output_dtype=output_dtype, backend=backend
            )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = int8_gemm(
            *tensors, group_size=256, output_dtype=output_dtype, backend=backend
        )
    graph.replay()
    assert_matches_reference(actual, tensors, 256)

    a, b, scale_a, scale_b = tensors
    a.fill_(1)
    b.fill_(-2)
    scale_a.fill_(0.125)
    scale_b.fill_(-0.25)
    graph.replay()
    expected = torch.full((256, 128), 48.0, dtype=output_dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("backend", ["auto", "triton", "cuda"])
def test_nondefault_stream_observes_prior_writes(backend):
    if backend == "cuda":
        require_native_cuda()
    output_dtype = torch.bfloat16 if backend == "cuda" else torch.float32
    tensors = inputs(256, 128, 768, 256, device="cuda", layout="column_major_b")
    # Warm the same launch outside the stream ordering check so compilation is
    # not responsible for establishing the producer/consumer dependency.
    int8_gemm(*tensors, group_size=256, output_dtype=output_dtype, backend=backend)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    complete = torch.cuda.Event()
    with torch.cuda.stream(stream):
        a, b, scale_a, scale_b = tensors
        a.fill_(-7)
        b.fill_(5)
        scale_a.fill_(0.125)
        scale_b.fill_(0.25)
        actual = int8_gemm(
            *tensors, group_size=256, output_dtype=output_dtype, backend=backend
        )
        complete.record()
    torch.cuda.current_stream().wait_event(complete)
    expected = torch.full((256, 128), -840.0, dtype=output_dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.fixture(
    scope="module",
    params=[("random", 256), ("random", 4352), ("signed_scales_and_extrema", 768)],
    ids=["minimum_k", "large_k", "signed_scales_and_extrema"],
)
def native_case(request):
    require_native_cuda()
    pattern, k = request.param
    if pattern == "random":
        tensors = inputs(256, 128, k, 256, device="cuda", layout="column_major_b")
    else:
        # Exact endpoint products, with different signs and zero scales across
        # rows, columns, and groups, expose wrong scale indexing or stale sums.
        rows = torch.arange(256)[:, None]
        cols = torch.arange(128)[:, None]
        groups = torch.arange(3)[None, :]
        endpoints = torch.tensor([-128, 127], dtype=torch.int8)
        a = endpoints[(rows + groups) % 2].repeat_interleave(256, dim=1).cuda()
        b = endpoints[(cols + groups // 2) % 2].repeat_interleave(256, dim=1).cuda().t()
        scale_a = torch.tensor([0, 0.125, -0.5, 1])[(rows + 2 * groups) % 4].cuda()
        scale_b = torch.tensor([-0.25, 0, 0.5, 2])[(cols + 3 * groups) % 4]
        scale_b = scale_b.t().contiguous().cuda()
        tensors = a, b, scale_a, scale_b
    expected = reference(*tensors, 256).to(torch.bfloat16)
    return tensors, expected, pattern == "signed_scales_and_extrema"


@cuda
@pytest.mark.parametrize("config_id", range(8))
def test_native_cuda_configurations(native_case, config_id):
    native_cuda = require_native_cuda()
    tensors, expected, exact = native_case
    actual = torch.empty((256, 128), device="cuda", dtype=torch.bfloat16)
    native_cuda.launch(*tensors, actual, config_id=config_id)
    if exact:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.cpu(), expected)


@cuda
@pytest.mark.parametrize("config_id,m,n", [(0, 192, 192), (7, 384, 256)])
@pytest.mark.parametrize("k", [256, 4352])
def test_native_cuda_partial_cta_bands(config_id, m, n, k):
    native_cuda = require_native_cuda()
    tensors = inputs(m, n, k, 256, device="cuda", layout="column_major_b")
    actual = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    native_cuda.launch(*tensors, actual, config_id=config_id)
    assert_matches_reference(actual, tensors, 256)


@cuda
@pytest.mark.parametrize("index", [0, 1])
def test_unaligned_storage_offset_rejects_native_and_falls_back(index):
    require_native_cuda()
    tensors = list(inputs(256, 128, 768, 256, device="cuda", layout="column_major_b"))
    source = tensors[index]
    backing = torch.empty(source.numel() + 1, device="cuda", dtype=torch.int8)[1:]
    view = backing.view(source.shape) if index == 0 else backing.view(128, 768).t()
    view.copy_(source)
    tensors[index] = view
    assert view.data_ptr() % 16 == 1
    assert (view if index == 0 else view.t()).is_contiguous()

    with pytest.raises(ValueError, match="aligned"):
        int8_gemm(*tensors, group_size=256, output_dtype=torch.bfloat16, backend="cuda")

    config = get_kernel_config(
        tensors[0], tensors[1], group_size=256, output_dtype=torch.bfloat16,
        scale_a=tensors[2], scale_b=tensors[3], backend="auto",
    )
    assert config["backend"] == "triton"
    actual = int8_gemm(*tensors, group_size=256, output_dtype=torch.bfloat16)
    assert_matches_reference(actual, tensors, 256)
