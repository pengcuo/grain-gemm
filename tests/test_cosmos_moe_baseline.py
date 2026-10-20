"""Check the real, optional upstream Cosmos BF16 grouped expert baseline.

Set GRAIN_COSMOS_DIR to a Cosmos Framework checkout to enable these checks.
The CPU reference evaluates each token/slot separately and never uses the
upstream histogram, permutation, padding, or grouped matrix multiplication.
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
def baseline():
    if not os.environ.get("GRAIN_COSMOS_DIR"):
        pytest.skip("Set GRAIN_COSMOS_DIR to enable the optional upstream baseline tests")
    path = Path(__file__).resolve().parents[1] / "benchmarks/cosmos_moe_baseline.py"
    spec = importlib.util.spec_from_file_location("_grain_test_cosmos_moe_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cosmos_root():
    value = os.environ.get("GRAIN_COSMOS_DIR")
    if not value:
        pytest.skip("Set GRAIN_COSMOS_DIR to enable the optional upstream baseline tests")
    root = Path(value).expanduser().resolve()
    assert (root / "cosmos_framework").is_dir(), f"Invalid GRAIN_COSMOS_DIR: {root}"
    return root


def inputs():
    generator = torch.Generator().manual_seed(20261020)
    experts, hidden, intermediate, tokens, top_k = 5, 256, 256, 7, 3
    x = torch.randn((tokens, hidden), generator=generator).to(torch.bfloat16)
    w1 = (torch.randn((experts, 2 * intermediate, hidden), generator=generator) / hidden**0.5).to(
        torch.bfloat16
    )
    w2 = (
        torch.randn((experts, hidden, intermediate), generator=generator) / intermediate**0.5
    ).to(torch.bfloat16)
    # Experts 3 and 4 are empty; repeated selections must each contribute.
    ids = torch.tensor(
        [[0, 1, 0], [2, 0, 1], [1, 2, 2], [0, 2, 1], [2, 2, 0], [1, 0, 2], [0, 1, 2]],
        dtype=torch.int64,
    )
    scores = torch.rand((tokens, top_k), generator=generator)
    # Match the real router: normalize in FP32, then round to BF16. The common
    # public comparison input stores those exact BF16 numeric values in FP32.
    scores = (scores / scores.sum(-1, keepdim=True)).to(torch.bfloat16).float()
    mask = torch.tensor([True, False, True, True, True, False, True])
    return x, w1, w2, ids, scores, mask


def load(baseline, cosmos_root, w1, w2, device):
    return baseline.load_cosmos_experts(
        cosmos_root,
        e=w1.shape[0],
        h=w1.shape[2],
        i=w2.shape[2],
        r=3,
        device=device,
        dtype=torch.bfloat16,
        w1_linear=w1,
        w2_linear=w2,
    )


def expected_counts(ids, experts, mask=None):
    result = [0] * experts
    for token, choices in enumerate(ids.tolist()):
        if mask is None or bool(mask[token]):
            for expert in choices:
                result[expert] += 1
    return torch.tensor(result, dtype=torch.int32)


def reference(x, w1, w2, ids, scores, mask):
    """Cosmos eager BF16 rounding points, with the score before FC2."""
    assert all(value.device.type == "cpu" for value in (x, w1, w2, ids, scores))
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        if mask is not None and not bool(mask[token]):
            continue
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            projected = (x[token : token + 1].float() @ w1[expert].float().t()).to(x.dtype)
            gate, up = projected.chunk(2, dim=-1)
            activated_gate = torch.nn.functional.silu(gate.float()).to(x.dtype)
            activated = (activated_gate.float() * up.float()).to(x.dtype)
            weighted = (activated.float() * scores[token, slot].to(x.dtype).float()).to(x.dtype)
            down = (weighted.float() @ w2[expert].float().t()).to(x.dtype)
            output[token] += down[0].float()
    return output.to(x.dtype)


def test_original_source_layout_metadata_and_counts_on_cpu(baseline, cosmos_root):
    _, w1, w2, ids, _, mask = inputs()
    module, metadata = load(baseline, cosmos_root, w1, w2, "cpu")
    source_dir = cosmos_root / "cosmos_framework/model/generator/reasoner/qwen3_vl_moe"
    source_path = source_dir / "moe.py"
    counts_path = source_dir / "qwen3_vl_moe.py"

    assert Path(module.forward.__code__.co_filename).resolve() == source_path
    assert Path(module._cosmos_weighted_expert_counts.__code__.co_filename).resolve() == counts_path
    # Compare the actual loaded class's syntax to the upstream definition, not
    # a separately maintained implementation or an assumed loader mode.
    original = next(
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == type(module).__name__
    )
    actual = ast.parse(inspect.getsource(type(module))).body[0]
    assert ast.dump(actual, include_attributes=False) == ast.dump(original, include_attributes=False)
    assert metadata["cosmos_dir"] == str(cosmos_root)
    assert metadata["token_group_alignment"] == 16
    assert metadata["score_position"] == "after SwiGLU, before down grouped GEMM"
    for relative, digest in metadata["source_sha256"].items():
        assert digest == hashlib.sha256((cosmos_root / relative).read_bytes()).hexdigest()

    assert module.gate_up_proj.stride() == (256 * 512, 512, 1)
    assert module.down_proj.stride() == (256 * 256, 256, 1)
    torch.testing.assert_close(module.gate_up_proj, w1.transpose(-2, -1), rtol=0, atol=0)
    torch.testing.assert_close(module.down_proj, w2.transpose(-2, -1), rtol=0, atol=0)
    for current_mask in (None, mask):
        weights = None if current_mask is None else current_mask.float()
        actual_counts = module._cosmos_weighted_expert_counts(ids, 5, weights)
        torch.testing.assert_close(actual_counts, expected_counts(ids, 5, current_mask), rtol=0, atol=0)
        assert actual_counts.dtype == torch.int32
        assert actual_counts[3:].tolist() == [0, 0]


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("masked", [False, True], ids=["all_real", "masked"])
def test_grouped_ffn_matches_independent_token_reference(baseline, cosmos_root, compiled, masked):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for upstream Cosmos grouped matrix multiplication")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("This PyTorch version does not provide torch._grouped_mm")
    x, w1, w2, ids, scores, mask = inputs()
    mask = mask if masked else None
    if mask is not None:
        # Masked values must never reach an expert. Multiplying a NaN result by
        # a zero routing score would fail to provide the same behavior.
        x[~mask] = float("nan")
    expected = reference(x, w1, w2, ids, scores, mask)
    module, _ = load(baseline, cosmos_root, w1, w2, "cuda")
    torch.testing.assert_close(module.gate_up_proj.cpu(), w1.transpose(-2, -1), rtol=0, atol=0)
    torch.testing.assert_close(module.down_proj.cpu(), w2.transpose(-2, -1), rtol=0, atol=0)
    assert module.gate_up_proj.stride() == (256 * 512, 512, 1)
    assert module.down_proj.stride() == (256 * 256, 256, 1)
    run = baseline.make_cosmos_forward(module, None if mask is None else mask.cuda())
    if compiled:
        run = torch.compile(run, fullgraph=True, dynamic=True)
    with torch.no_grad():
        actual = run(x.cuda(), ids.cuda(), scores.cuda()).cpu()
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    if mask is not None:
        torch.testing.assert_close(actual[~mask], torch.zeros_like(actual[~mask]), rtol=0, atol=0)
    # Inductor can fuse the BF16 pointwise operations, and BF16 scatter_add has
    # a different addition order from the per-slot FP32 reference. Bound those
    # rounding differences while retaining an independent numerical check.
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1e-12)
    assert relative_l2 < 0.015, f"relative L2 error: {relative_l2.item():.6f}"
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
