"""Build the original vLLM CUDA operators needed by its BF16 Triton MoE.

This optional benchmark helper compiles the upstream CUDA files unchanged;
only the three-operator registration unit is supplied here. No vLLM package
installation, alternate alignment algorithm, or floating-point fallback is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import types


_BINDINGS = r"""
#include <torch/csrc/stable/library.h>
#include "moe/moe_ops.h"

void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);

STABLE_TORCH_LIBRARY_FRAGMENT(NATIVE_NAMESPACE, m) {
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.def("moe_sum(Tensor input, Tensor! output, Tensor? topk_ids=None, "
        "Tensor? expert_map=None) -> ()");
  m.def("moe_align_block_size(Tensor topk_ids, int num_experts, "
        "int block_size, Tensor! sorted_token_ids, Tensor! experts_ids, "
        "Tensor! num_tokens_post_pad, Tensor? maybe_expert_map, "
        "Tensor(a!)? scatter_idx=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(NATIVE_NAMESPACE, CUDA, m) {
  m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  m.impl("moe_sum", TORCH_BOX(&moe_sum));
  m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_dependencies(csrc: Path, entrypoints: list[Path]) -> dict[str, str]:
    """Record the original sources and recursively included local headers."""
    pending = list(entrypoints)
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), re.MULTILINE):
            for candidate in (path.parent / include, csrc / include):
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(csrc):
                        raise ValueError(f"Upstream include escapes csrc: {resolved}")
                    pending.append(resolved)
                    break
            else:
                raise FileNotFoundError(f"Cannot resolve upstream include {include!r} in {path}")
    return {str(path.relative_to(csrc.parent)): _sha256(path) for path in sorted(visited)}


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def load_native_ops(
    vllm_dir: str | Path,
    *,
    arch: str = "11.0a",
    verbose: bool = False,
) -> tuple[types.SimpleNamespace, dict]:
    """Return the three original CUDA ops and an exact build/source manifest.

    ``arch`` is explicit so compiling does not query or initialize a GPU. Build
    products live in the user cache, outside the grain-gemm checkout. Runtime
    calls use the current PyTorch CUDA stream through the upstream stable ABI.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not re.fullmatch(r"[0-9]+\.[0-9]+a?", arch):
        raise ValueError("arch must be a CUDA capability such as '11.0a'")
    if CUDA_HOME is None:
        raise RuntimeError("The standalone vLLM baseline requires a CUDA toolkit with nvcc")
    root = Path(vllm_dir).expanduser().resolve()
    csrc = root / "csrc"
    sources = [csrc / "moe/moe_align_sum_kernels.cu", csrc / "activation_kernels.cu"]
    dependencies = _upstream_dependencies(csrc, sources + [csrc / "moe/moe_ops.h"])
    capability = arch.replace(".", "")
    cflags = [
        "-O3", "-DNDEBUG", "-DUSE_CUDA", "-std=c++20", "-fvisibility=hidden",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
    ]
    cuda_flags = [
        "-O3", "-DNDEBUG", "-DUSE_CUDA", "-std=c++20",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Xcompiler=-fvisibility=hidden",
        f"-gencode=arch=compute_{capability},code=sm_{capability}",
    ]
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    if cuda_version >= (11, 8):
        cuda_flags.append("-DENABLE_FP8")
    if cuda_version >= (12, 0):
        # Match cmake/utils.cmake: cpp_extension adds these COMMON_NVCC_FLAGS,
        # but upstream removes them when compiling with CUDA 12 or newer.
        cuda_flags += [
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ]
    nvcc = Path(CUDA_HOME) / "bin/nvcc"
    nvcc_version = subprocess.check_output([str(nvcc), "--version"], text=True).strip()
    descriptor = dict(
        original_source_sha256=dependencies,
        upstream_build_sha256={
            path: _sha256(root / path) for path in ("CMakeLists.txt", "cmake/utils.cmake")
        },
        binding_template_sha256=hashlib.sha256(_BINDINGS.encode()).hexdigest(),
        arch=arch,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        nvcc_version=nvcc_version,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
    )
    signature = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:20]
    name = "grain_vllm_moe_" + signature
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    build_dir = (cache_root / "grain-gemm/vllm-moe-native" / signature).resolve()
    checkout = Path(__file__).resolve().parents[1]
    if build_dir.is_relative_to(checkout):
        raise ValueError("The native benchmark build cache must be outside the grain-gemm checkout")
    build_dir.mkdir(parents=True, exist_ok=True)
    binding = build_dir / "bindings.cpp"
    binding_text = _BINDINGS.replace("NATIVE_NAMESPACE", name)
    if not binding.exists() or binding.read_text() != binding_text:
        binding.write_text(binding_text)
    library = Path(load(
        name=name,
        sources=[str(binding), *(str(path) for path in sources)],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(csrc)],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=verbose,
    ))
    registered = getattr(torch.ops, name)

    def silu_and_mul(out, inputs):
        registered.silu_and_mul(out, inputs)

    def moe_sum(inputs, output, topk_ids=None, expert_map=None):
        registered.moe_sum(inputs, output, topk_ids, expert_map)

    def moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_token_ids, experts_ids,
        num_tokens_post_pad, expert_map=None, scatter_idx=None,
    ):
        registered.moe_align_block_size(
            topk_ids, num_experts, block_size, sorted_token_ids, experts_ids,
            num_tokens_post_pad, expert_map, scatter_idx,
        )

    metadata = dict(
        **descriptor,
        upstream_dir=str(root),
        upstream_commit=_git_output(root, "rev-parse", "HEAD"),
        upstream_remote=_git_output(root, "remote", "get-url", "origin"),
        upstream_source_dirty=bool(_git_output(root, "status", "--porcelain", "--", *dependencies)),
        namespace=name,
        build_directory=str(build_dir),
        generated_binding_sha256=_sha256(binding),
        generated_binding_source=binding_text,
        library_path=str(library),
        library_sha256=_sha256(library),
        build_ninja_sha256=_sha256(build_dir / "build.ninja"),
        standalone_scope="Unchanged moe_align_sum_kernels.cu and activation_kernels.cu; new registration only",
        algorithm="Original upstream CUDA dispatcher, including small-batch and generic expert alignment paths",
    )
    (build_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return types.SimpleNamespace(
        silu_and_mul=silu_and_mul,
        moe_sum=moe_sum,
        moe_align_block_size=moe_align_block_size,
    ), metadata
