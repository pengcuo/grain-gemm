"""Check the extracted kernel against an independent, exact INT32 dot product."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm.baselines import sglang_int8_gemm


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def reference(a, b, scale_a, scale_b, group_size):
    """Compute each integer dot product on the CPU before applying group scales."""
    a, b = a.cpu().to(torch.int32), b.cpu().to(torch.int32)
    scale_a, scale_b = scale_a.cpu(), scale_b.cpu()
    result = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], group_size)):
        partial = a[:, start : start + group_size] @ b[start : start + group_size]
        scales = scale_a[:, group, None] * scale_b[group, None, :]
        result += partial.to(torch.float32) * scales
    return result


def inputs(m, n, k, group_size, *, device="cpu", transposed=False):
    generator = torch.Generator(device=device).manual_seed(2026)
    groups = (k + group_size - 1) // group_size

    def integers(shape):
        return torch.randint(
            -128, 128, shape, dtype=torch.int8, device=device, generator=generator
        )

    def scales(shape):
        return 0.001 + torch.rand(shape, device=device, generator=generator) * 0.05

    if transposed:
        return (
            integers((k, m)).t(),
            integers((n, k)).t(),
            scales((groups, m)).t(),
            scales((n, groups)).t(),
        )
    return integers((m, k)), integers((k, n)), scales((m, groups)), scales((groups, n))


@cuda
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_groups_with_m_n_k_tails(group_size):
    tensors = inputs(19, 37, 2 * group_size + 13, group_size, device="cuda")
    actual = sglang_int8_gemm(*tensors, group_size=group_size)
    expected = reference(*tensors, group_size)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-3)


@cuda
def test_transposed_inputs_and_scales():
    tensors = inputs(33, 65, 269, 128, device="cuda", transposed=True)
    assert all(not tensor.is_contiguous() for tensor in tensors)
    actual = sglang_int8_gemm(*tensors, group_size=128)
    torch.testing.assert_close(
        actual.cpu(), reference(*tensors, 128), rtol=1e-5, atol=1e-3
    )


@cuda
def test_distinct_scales_for_every_row_column_and_group():
    group_size = 32
    a = torch.ones((3, 69), dtype=torch.int8, device="cuda")
    b = torch.cat(
        [
            torch.full((length, 4), value, dtype=torch.int8)
            for length, value in [(32, 1), (32, -2), (5, 3)]
        ]
    ).cuda()
    scale_a = torch.tensor([[1, 8, 0.25], [2, 4, 1], [0.5, 2, 4]], device="cuda")
    scale_b = torch.tensor([[0.5, 2, 8, 1], [4, 1, 0.25, 2], [2, 0.5, 1, 4]], device="cuda")
    actual = sglang_int8_gemm(a, b, scale_a, scale_b, group_size=group_size)
    torch.testing.assert_close(
        actual.cpu(), reference(a, b, scale_a, scale_b, group_size), rtol=0, atol=0
    )


@cuda
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_output_dtype(output_dtype):
    tensors = inputs(17, 29, 103, 64, device="cuda")
    actual = sglang_int8_gemm(*tensors, group_size=64, output_dtype=output_dtype)
    expected = reference(*tensors, 64).to(output_dtype)
    assert actual.dtype == output_dtype
    torch.testing.assert_close(actual.cpu(), expected)


@cuda
@pytest.mark.parametrize("shape", [(0, 7, 65), (7, 0, 65), (7, 9, 0)])
def test_empty_dimensions(shape):
    m, n, k = shape
    actual = sglang_int8_gemm(*inputs(m, n, k, 32, device="cuda"), group_size=32)
    assert actual.shape == (m, n)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cuda"
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize(
    "options",
    [
        {"group_size": 16},
        {"block_m": 24},
        {"block_n": 8},
        {"num_warps": 3},
        {"num_stages": 0},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        sglang_int8_gemm(*inputs(2, 3, 32, 256), **options)


@pytest.mark.parametrize(
    "index,replacement,exception",
    [
        (0, lambda t: t.float(), TypeError),
        (1, lambda t: t.to(torch.int16), TypeError),
        (2, lambda t: t.half(), TypeError),
        (3, lambda t: t.half(), TypeError),
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
        sglang_int8_gemm(*tensors)


def test_invalid_output_dtype():
    with pytest.raises(TypeError):
        sglang_int8_gemm(*inputs(2, 3, 32, 256), output_dtype=torch.int32)


def test_cpu_inputs_are_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        sglang_int8_gemm(*inputs(2, 3, 32, 256))
