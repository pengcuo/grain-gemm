"""Load the upstream Cosmos grouped-MM expert FFN without changing its kernels.

The caller owns inputs, quantization, compilation, timing, and CUDA Graph capture.
Only weight conversion happens at construction; every forward includes the GPU
expert histogram, upstream dispatch/padding, both grouped GEMMs, and combine.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


_SOURCE_DIR = Path("cosmos_framework/model/generator/reasoner/qwen3_vl_moe")
_EXTRACTED_NAMES = ("_run_experts_grouped_mm", "Qwen3VLMoeTextExpertsGroupedMm")
_MODULE_NAME = "cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe"


class _EquivalentSiluActivation(nn.Module):
    """The same F.silu operation as transformers 4.57.6 SiLUActivation.forward."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs)


def _activation_provider() -> tuple[object, dict]:
    try:
        from transformers.activations import ACT2FN

        return ACT2FN, {"activation_provider": "transformers.activations.ACT2FN"}
    except ImportError as exc:
        # Only this benchmark's explicitly selected silu activation is supplied.
        # There is no generic fallback for any other upstream activation name.
        return {"silu": _EquivalentSiluActivation()}, {
            "activation_provider": "equivalent PyTorch silu provider",
            "activation_source_url": (
                "https://github.com/huggingface/transformers/blob/v4.57.6/"
                "src/transformers/activations.py"
            ),
            "activation_equivalence": "SiLUActivation.forward calls nn.functional.silu(input)",
            "activation_import_error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_metadata(root: Path) -> dict:
    paths = [
        _SOURCE_DIR / name
        for name in (
            "moe.py",
            "moe_kernels.py",
            "qwen3_vl_moe.py",
            "configuration_qwen3_vl_moe.py",
            "moe_bench.py",
        )
    ]
    result = {
        "cosmos_dir": str(root),
        "source_sha256": {str(path): _sha256(root / path) for path in paths},
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        result["git_remote"] = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        result["source_dirty"] = bool(
            subprocess.check_output(
                [
                    "git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--",
                    *(str(path) for path in paths),
                ],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        result["git_error"] = str(exc)
    return result


def _load_kernel_module(path: Path):
    # Triton inspects the original Python file; do not copy or AST-transform it.
    name = "_grain_cosmos_moe_kernels_" + _sha256(path)[:16]
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream kernel module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_counts_function(path: Path) -> Callable:
    """Extract the unchanged router histogram helper without importing the model."""
    source = ast.parse(path.read_text(), filename=str(path))
    definitions = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_weighted_expert_counts"
    ]
    if len(definitions) != 1:
        raise ImportError(f"Expected one _weighted_expert_counts definition in {path}")
    name = "_grain_cosmos_moe_counts_" + _sha256(path)[:16]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.torch = torch
    sys.modules[name] = module
    extracted = ast.Module(body=definitions, type_ignores=[])
    exec(compile(extracted, str(path), "exec", dont_inherit=True), module.__dict__)
    return module._weighted_expert_counts


def _load_experts_class(root: Path) -> tuple[type[nn.Module], dict]:
    activations, activation_metadata = _activation_provider()

    source_path = root / _SOURCE_DIR / "moe.py"
    existing = sys.modules.get(_MODULE_NAME)
    normal_import_error = None
    if activation_metadata["activation_provider"] != "transformers.activations.ACT2FN":
        normal_import_error = activation_metadata["activation_import_error"]
    elif existing is None or Path(existing.__file__).resolve() == source_path:
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module(_MODULE_NAME)
            if Path(module.__file__).resolve() != source_path:
                raise ImportError(f"Imported another Cosmos checkout: {module.__file__}")
            return module.Qwen3VLMoeTextExpertsGroupedMm, {
                "loader": "normal_import",
                "module": _MODULE_NAME,
                "extracted_names": [],
                "token_group_alignment": int(module.TOKEN_GROUP_ALIGN_SIZE_M),
                **activation_metadata,
            }
        except ImportError as exc:
            normal_import_error = f"{type(exc).__name__}: {exc}"
        finally:
            sys.path.remove(str(root))
    else:
        normal_import_error = f"Another Cosmos checkout is already imported: {existing.__file__}"

    # Keep the upstream function/class AST nodes unchanged. Only their unrelated
    # framework imports and config annotation are supplied by this namespace.
    parsed = ast.parse(source_path.read_text(), filename=str(source_path))
    definitions = {
        node.name: node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _EXTRACTED_NAMES
    }
    if set(definitions) != set(_EXTRACTED_NAMES):
        raise ImportError(f"Expected upstream definitions {_EXTRACTED_NAMES} in {source_path}")
    kernels = _load_kernel_module(root / _SOURCE_DIR / "moe_kernels.py")
    name = "_grain_cosmos_moe_experts_" + _sha256(source_path)[:16]
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    module.__dict__.update(
        torch=torch,
        nn=nn,
        Callable=Callable,
        ACT2FN=activations,
        Qwen3VLMoeTextConfig=object,
        TOKEN_GROUP_ALIGN_SIZE_M=kernels.TOKEN_GROUP_ALIGN_SIZE_M,
        _generate_permute_indices=kernels._generate_permute_indices,
    )
    extracted = ast.Module(body=[definitions[name] for name in _EXTRACTED_NAMES], type_ignores=[])
    sys.modules[name] = module
    exec(compile(extracted, str(source_path), "exec", dont_inherit=True), module.__dict__)
    return module.Qwen3VLMoeTextExpertsGroupedMm, {
        "loader": "unchanged_ast_definitions_and_original_kernel_module",
        "module": name,
        "extracted_names": list(_EXTRACTED_NAMES),
        "normal_import_error": normal_import_error,
        "kernel_module": kernels.__name__,
        "token_group_alignment": int(kernels.TOKEN_GROUP_ALIGN_SIZE_M),
        **activation_metadata,
    }


def load_cosmos_experts(
    cosmos_dir: str | Path,
    e: int,
    h: int,
    i: int,
    r: int,
    device: torch.device | str,
    dtype: torch.dtype,
    w1_linear: torch.Tensor,
    w2_linear: torch.Tensor,
) -> tuple[nn.Module, dict]:
    """Construct the unchanged upstream experts with its original N-contiguous weights.

    ``w1_linear`` is ``[E,2I,H]`` with gate channels first; ``w2_linear``
    is ``[E,H,I]``. Their transpose/copy is performed once, outside timing.
    The returned module has the original upstream five-argument forward.
    """
    if min(e, h, i, r) <= 0 or r > e:
        raise ValueError("Require positive E/H/I/R and R <= E")
    if tuple(w1_linear.shape) != (e, 2 * i, h) or tuple(w2_linear.shape) != (e, h, i):
        raise ValueError("Expected Linear weights [E,2I,H] and [E,H,I]")
    root = Path(cosmos_dir).expanduser().resolve()
    metadata = _source_metadata(root)
    cls, loader = _load_experts_class(root)
    counts = _load_counts_function(root / _SOURCE_DIR / "qwen3_vl_moe.py")
    config = types.SimpleNamespace(
        num_experts=e,
        hidden_size=h,
        moe_intermediate_size=i,
        num_experts_per_tok=r,
        hidden_act="silu",
    )
    # Meta construction preserves the upstream strides without an intermediate
    # full-size FP32 allocation on the GPU.
    with torch.device("meta"):
        module = cls(config, top_k=r)
    module = module.to(dtype=dtype).to_empty(device=device)
    with torch.no_grad():
        module.gate_up_proj.copy_(w1_linear.transpose(-2, -1).contiguous())
        module.down_proj.copy_(w2_linear.transpose(-2, -1).contiguous())
    module.eval().requires_grad_(False)
    module._cosmos_weighted_expert_counts = counts
    metadata.update(loader)
    metadata["extracted_names"].append("_weighted_expert_counts")
    metadata.update(
        dtype=str(dtype),
        weight_layout="upstream N-contiguous Parameters; Linear transpose/copy once before timing",
        gate_up_shape=list(module.gate_up_proj.shape),
        gate_up_stride=list(module.gate_up_proj.stride()),
        down_shape=list(module.down_proj.shape),
        down_stride=list(module.down_proj.stride()),
        score_position="after SwiGLU, before down grouped GEMM",
        counts=(
            "unchanged upstream _weighted_expert_counts, included in timed forward; "
            "no mask: torch.histc int32 input; mask: int32 scatter_add"
        ),
        counts_loader="unchanged AST definition from qwen3_vl_moe.py",
        score_dtype="cast to hidden_states.dtype inside timed forward",
    )
    return module, metadata


def cosmos_forward(
    module: nn.Module,
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run counts and the original Cosmos dispatch/grouped-FFN/combine each call.

    Router logits/top-k selection are inputs and are excluded for both methods.
    For the BF16 comparison the caller first rounds scores to BF16, then passes
    those identical numeric values to INT8 as FP32. This cast back to activation
    dtype is intentionally inside the BF16 timed call.
    """
    token_weight = None if token_mask is None else token_mask.to(torch.float32)
    counts = module._cosmos_weighted_expert_counts(ids, module.num_experts, token_weight)
    if token_mask is None:
        active_scores = scores.to(x.dtype)
    else:
        active_scores = torch.where(token_mask[:, None], scores.to(x.dtype), 0)
    return module(x, active_scores, ids, counts, token_mask)


def make_cosmos_forward(
    module: nn.Module,
    token_mask: torch.Tensor | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return ``forward(x, ids, scores)`` for caller-controlled torch.compile."""
    def run(x: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return cosmos_forward(module, x, ids, scores, token_mask)

    return run
