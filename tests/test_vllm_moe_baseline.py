"""Check the optional, original vLLM BF16 expert FFN used by Omni.

Set GRAIN_VLLM_DIR to an upstream vLLM checkout. The numerical checks compare
the original GPU route/activation/GEMM/reduction pipeline with an independent
CPU token/slot reference, including duplicate choices and empty experts.
"""

import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture(scope="module")
def vllm_root():
    value = os.environ.get("GRAIN_VLLM_DIR")
    if not value:
        pytest.skip("Set GRAIN_VLLM_DIR to enable the optional upstream baseline tests")
    root = Path(value).expanduser().resolve()
    assert (root / "vllm/model_executor/layers/fused_moe/fused_moe.py").is_file(), (
        f"Invalid GRAIN_VLLM_DIR: {root}"
    )
    return root


@pytest.fixture(scope="module")
def baseline(vllm_root):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the original vLLM BF16 Triton FFN")
    benchmark_dir = Path(__file__).resolve().parents[1] / "benchmarks"
    sys.path.insert(0, str(benchmark_dir))
    try:
        modules = []
        for filename in ("vllm_moe_baseline", "bench_moe_vllm"):
            spec = importlib.util.spec_from_file_location(
                "_grain_test_" + filename, benchmark_dir / (filename + ".py")
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        runner, metadata = modules[0].load_vllm_experts(vllm_root)
        yield runner, metadata, modules[1].vllm_cpu_reference
    finally:
        sys.path.remove(str(benchmark_dir))


def inputs(id_dtype):
    generator = torch.Generator().manual_seed(20261020)
    experts, tokens, hidden, intermediate, top_k = 8, 17, 128, 64, 4
    x = torch.randn((tokens, hidden), generator=generator).bfloat16()
    w1 = (
        torch.randn((experts, 2 * intermediate, hidden), generator=generator) / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn((experts, hidden, intermediate), generator=generator) / intermediate**0.5
    ).bfloat16()
    ids = torch.randint(0, 3, (tokens, top_k), generator=generator, dtype=id_dtype)
    # This token has the same expert in every slot. Each score must contribute;
    # experts 3..7 receive no routes, and expert counts are not tile-aligned.
    ids[0] = 2
    ids[1] = torch.tensor([0, 1, 0, 2], dtype=id_dtype)
    scores = torch.rand((tokens, top_k), generator=generator) * 1.7
    # Deliberately retain nonnormalized FP32 scores: hidden normalization or
    # dropping a repeated expert would change the requested FFN.
    scores[0] = torch.tensor([0.0, 0.13, 0.57, 1.31])
    scores[1] = torch.tensor([0.0, 0.0, 0.0, 0.0])
    return x, w1, w2, scores, ids


def test_loaded_ffn_and_native_operators_are_original_sources(baseline, vllm_root):
    runner, metadata, _ = baseline
    module = runner.module
    paths_and_names = {
        "fused_moe.py": (
            "write_zeros_to_output", "fused_moe_kernel", "invoke_fused_moe_triton_kernel",
            "dispatch_fused_moe_kernel", "try_get_optimal_moe_config",
            "_prepare_expert_assignment", "fused_experts_impl",
        ),
        "moe_align_block_size.py": ("moe_align_block_size",),
        "activation.py": ("apply_moe_activation",),
        "utils.py": ("moe_kernel_quantize_input", "resolve_moe_use_td"),
    }
    source_dir = vllm_root / "vllm/model_executor/layers/fused_moe"
    for filename, names in paths_and_names.items():
        source_path = source_dir / filename
        definitions = ast.parse(source_path.read_text()).body
        for name in names:
            function = getattr(module, name)
            function = getattr(function, "fn", function)  # Triton JITFunction
            assert Path(function.__code__.co_filename).resolve() == source_path
            original = next(
                node for node in definitions
                if isinstance(node, ast.FunctionDef) and node.name == name
                and not any(
                    isinstance(decorator, ast.Name) and decorator.id == "overload"
                    for decorator in node.decorator_list
                )
            )
            actual = ast.parse(inspect.getsource(function)).body[0]
            assert ast.dump(actual, include_attributes=False) == ast.dump(
                original, include_attributes=False
            ), name

    assert metadata["source_dirty"] is False
    assert metadata["upstream_entrypoint"] == "fused_experts_impl"
    assert module.resolve_moe_use_td() is False
    for relative, digest in metadata["source_sha256"].items():
        assert digest == hashlib.sha256((vllm_root / relative).read_bytes()).hexdigest()
    native = metadata["native"]
    assert native["arch"] == "11.0a"
    assert native["upstream_commit"] == metadata["git_commit"]
    assert native["upstream_source_dirty"] is False
    native_sources = native["original_source_sha256"]
    assert {"csrc/activation_kernels.cu", "csrc/moe/moe_align_sum_kernels.cu"} <= native_sources.keys()
    for relative, digest in native_sources.items():
        assert digest == hashlib.sha256((vllm_root / relative).read_bytes()).hexdigest()
    assert native["library_sha256"] == hashlib.sha256(Path(native["library_path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_original_ffn_preserves_duplicate_routes_and_empty_experts(baseline, id_dtype):
    runner, _, reference = baseline
    x, w1, w2, scores, ids = inputs(id_dtype)
    expected = reference(x, ids, scores, w1, w2, torch)
    with torch.inference_mode():
        actual = runner(x.cuda(), w1.cuda(), w2.cuda(), scores.cuda(), ids.cuda()).cpu()
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_l2 < 0.01, f"relative L2 error: {relative_l2.item():.6f}"
    # FP64 CPU dots and FP32 tensor-core accumulation may differ at a BF16
    # rounding boundary. A broken route or changed rounding policy produces a
    # much larger full-tensor error than these bounded numerical differences.
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.015)
