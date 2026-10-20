"""Load the pinned, original vLLM BF16 Triton expert FFN for benchmarking.

Only serving/configuration imports are isolated. The upstream Python function
bodies, Triton kernels, CUDA alignment/activation/reduction, and tile selection
are retained. This is the pre-routed, single-GPU functional Triton path, not a
vLLM engine or a claim that every upstream backend is supported on Thor.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from enum import Enum
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Any, Literal

import torch
import triton
import triton.language as tl


PINNED_REVISION = "4e304b4d1ed0037514073305d839a79586d0e9de"
MOE_DIR = Path("vllm/model_executor/layers/fused_moe")
_CORE_NAMES = (
    "write_zeros_to_output", "fused_moe_kernel", "invoke_fused_moe_triton_kernel",
    "dispatch_fused_moe_kernel", "get_config_file_name", "get_moe_configs",
    "get_default_config", "try_get_optimal_moe_config", "_prepare_expert_assignment",
    "_get_config_quant_dtype", "fused_experts_impl",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unsupported(*args, **kwargs):
    raise NotImplementedError("This upstream loader supports unquantized BF16, standard SiLU, single GPU only")


class _Logger:
    def __init__(self):
        self._seen = set()
        self._logger = logging.getLogger(__name__)

    def _once(self, level, message, *args, **kwargs):
        key = (level, message, repr(args))
        if key not in self._seen:
            self._seen.add(key)
            self._logger.log(level, message, *args)

    def info_once(self, *args, **kwargs):
        self._once(logging.INFO, *args, **kwargs)

    def warning_once(self, *args, **kwargs):
        self._once(logging.WARNING, *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)


class _CudaPlatform:
    """Only platform queries reached by the selected single-CUDA BF16 path."""

    def __init__(self, name: str, capability: tuple[int, int]):
        self.name, self.capability = name, capability

    def get_device_name(self, device_id=0):
        if device_id != 0:
            raise NotImplementedError("The benchmark loader targets device 0")
        return self.name

    def is_rocm(self):
        return False

    def is_xpu(self):
        return False

    def is_cuda(self):
        return True

    def is_device_capability(self, capability):
        value = (capability // 10, capability % 10) if isinstance(capability, int) else capability
        return self.capability == value

    def fp8_dtype(self):
        # The original quantization helper compares against this before its
        # unquantized no-op branch. No FP8 operation is performed.
        return torch.float8_e4m3fn


class _ActivationTorch:
    """Keep upstream torch.ops call syntax without changing global torch.ops."""

    def __init__(self, native_ops):
        self.ops = types.SimpleNamespace(_C=types.SimpleNamespace(silu_and_mul=native_ops.silu_and_mul))

    def __getattr__(self, name):
        return getattr(torch, name)


def _definitions(path: Path, names: tuple[str, ...], namespace: dict, *, class_name=None):
    """Execute original AST nodes and retain their original source locations."""
    parsed = ast.parse(path.read_text(), filename=str(path))
    body = parsed.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = []
    found = set()
    for node in body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in names:
            # moe_align_block_size has typing overload declarations ahead of
            # its one runtime definition; execute only the runtime body.
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(decorator, ast.Name) and decorator.id == "overload"
                for decorator in node.decorator_list
            ):
                continue
            if name in found:
                raise ImportError(f"Duplicate runtime definition {name} in {path}")
            selected.append(node)
            found.add(name)
    if found != set(names):
        raise ImportError(f"Missing upstream definitions in {path}: {set(names) - found}")
    module_name = "_grain_vllm_" + hashlib.sha256((str(path) + repr(names)).encode()).hexdigest()[:20]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    sys.modules[module_name] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _source_metadata(root: Path):
    files = [MOE_DIR / name for name in (
        "fused_moe.py", "moe_align_block_size.py", "activation.py", "utils.py",
        "config.py", "modular_kernel.py",
    )] + [Path("vllm/utils/math_utils.py"), Path("vllm/utils/platform_utils.py")]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"Expected pinned vLLM {PINNED_REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", *(str(p) for p in files)],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"Upstream benchmark sources must be unchanged: {dirty}")
    return dict(
        vllm_dir=str(root), git_commit=revision,
        git_remote=subprocess.check_output(["git", "-C", str(root), "remote", "get-url", "origin"], text=True).strip(),
        source_dirty=False, source_sha256={str(path): _sha256(root / path) for path in files},
        loader_sha256=_sha256(Path(__file__)),
    )


def _load_core(root: Path, native_ops, *, device_name: str, capability=(11, 0)):
    """Dependency isolation; can be inspected on CPU using a native-op stub."""
    logger = _Logger()
    platform = _CudaPlatform(device_name, tuple(capability))
    td_value = os.environ.get("VLLM_TRITON_USE_TD")
    if td_value not in (None, "0"):
        raise NotImplementedError("Use the upstream default CUDA pointer path; VLLM_TRITON_USE_TD must be unset or 0")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        raise NotImplementedError("This comparison requires default VLLM_BATCH_INVARIANT=0")
    envs = types.SimpleNamespace(
        VLLM_TRITON_USE_TD=None if td_value is None else False,
        VLLM_BATCH_INVARIANT=False,
        VLLM_TUNED_CONFIG_FOLDER=os.environ.get("VLLM_TUNED_CONFIG_FOLDER"),
    )

    def import_without_serving(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.model_executor.layers.fused_moe" and tuple(fromlist) == ("get_config",):
            return types.SimpleNamespace(get_config=lambda: None)
        if name == "vllm.platforms" and tuple(fromlist) == ("current_platform",):
            return types.SimpleNamespace(current_platform=platform)
        if name == "vllm" or name.startswith("vllm."):
            raise NotImplementedError(f"Unsupported serving import in BF16 benchmark: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    common = dict(torch=torch, triton=triton, tl=tl, functools=functools, json=json, os=os, Any=Any,
                  envs=envs, current_platform=platform, logger=logger,
                  __builtins__={**vars(builtins), "__import__": import_without_serving})
    math_module = _definitions(root / "vllm/utils/math_utils.py", ("next_power_of_2", "round_up"), common)
    platform_module = _definitions(root / "vllm/utils/platform_utils.py", ("get_device_name_as_file_name",),
                                   {**common, "cache": functools.cache, "re": re})
    activations = _definitions(
        root / MOE_DIR / "activation.py",
        ("MoEActivation", "_STR_ALIASES", "_CUSTOM_OP_NAMES", "_WITHOUT_MUL",
         "ApplyMoEActivationConfig", "_DEFAULT_APPLY_MOE_ACTIVATION_CONFIG",
         "_validate_moe_activation_shapes", "apply_moe_activation"),
        {**common, "torch": _ActivationTorch(native_ops), "Enum": Enum, "dataclass": dataclass,
         "F": torch.nn.functional, "_apply_moe_activation_masked": _unsupported,
         "silu_and_mul_with_clamp": _unsupported},
    )
    dimensions = _definitions(
        root / MOE_DIR / "modular_kernel.py", ("adjust_N_for_activation",),
        {**common, "MoEActivation": activations.MoEActivation}, class_name="FusedMoEExpertsModular",
    )
    utils = _definitions(
        root / MOE_DIR / "utils.py",
        ("moe_kernel_quantize_input", "enable_swap_ab", "resolve_moe_use_td", "warn_if_moe_use_td_ineffective"),
        {**common, "_warned_moe_use_td_ineffective": False,
         **{name: _unsupported for name in (
             "_fp8_quantize_dequantize", "_fp8_quantize", "_int8_quantize",
             "_nvfp4_quantize", "ref_nvfp4_quant_dequant", "_mxfp4_quantize",
             "_mxfp8_e4m3_quantize", "_mxfp6_e3m2_quantize", "_mxfp6_e2m3_quantize",
         )}},
    )
    alignment = _definitions(
        root / MOE_DIR / "moe_align_block_size.py", ("moe_align_block_size",),
        {**common, "Literal": Literal, "ops": native_ops, "round_up": math_module.round_up},
    )
    config = _definitions(root / MOE_DIR / "config.py", ("_get_config_dtype_str",), common)
    module = _definitions(
        root / MOE_DIR / "fused_moe.py", _CORE_NAMES,
        {**common, "ops": native_ops, "MoEActivation": activations.MoEActivation,
         "apply_moe_activation": activations.apply_moe_activation,
         "mk": types.SimpleNamespace(FusedMoEExpertsModular=types.SimpleNamespace(
             adjust_N_for_activation=dimensions.adjust_N_for_activation)),
         "_get_config_dtype_str": config._get_config_dtype_str,
         "moe_align_block_size": alignment.moe_align_block_size,
         "enable_swap_ab": utils.enable_swap_ab,
         "moe_kernel_quantize_input": utils.moe_kernel_quantize_input,
         "resolve_moe_use_td": utils.resolve_moe_use_td,
         "warn_if_moe_use_td_ineffective": utils.warn_if_moe_use_td_ineffective,
         "next_power_of_2": math_module.next_power_of_2,
         "get_device_name_as_file_name": platform_module.get_device_name_as_file_name,
         "set_triton_allocator": _unsupported,
         "should_moe_wna16_use_cuda": _unsupported,
         "invoke_fused_moe_wna16_cuda_kernel": _unsupported,
         "invoke_fused_moe_wna16_triton_kernel": _unsupported},
    )
    return module, envs


class _Runner:
    def __init__(self, module, root, envs, metadata):
        self.module, self.root, self.envs, self.metadata = module, root, envs, metadata

    def config_for(self, tokens, w1, w2, top_k):
        """Record the original selector's result outside the timed call."""
        config = self.module.try_get_optimal_moe_config(tuple(w1.shape), tuple(w2.shape), top_k, None, tokens)
        filename = self.module.get_config_file_name(w2.shape[0], w2.shape[2], None)
        candidates = []
        if self.envs.VLLM_TUNED_CONFIG_FOLDER:
            candidates.append(Path(self.envs.VLLM_TUNED_CONFIG_FOLDER) / filename)
        candidates.append(self.root / MOE_DIR / "configs" / filename)
        found = next((path for path in candidates if path.exists()), None)
        self.metadata["configurations"][str(tokens)] = dict(
            w1_shape=list(w1.shape), w2_shape=list(w2.shape), top_k=top_k,
            config=dict(config), config_filename=filename,
            source="tuned_file" if found is not None else "upstream_get_default_config",
            config_path=None if found is None else str(found),
            config_sha256=None if found is None else _sha256(found),
        )
        return dict(config)

    def __call__(self, x, w1, w2, scores, ids):
        if x.dtype != torch.bfloat16 or w1.dtype != x.dtype or w2.dtype != x.dtype:
            raise ValueError("The vLLM baseline requires BF16 activations and weights")
        if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or ids.ndim != 2:
            raise ValueError("Expected x[T,H], w1[E,2I,H], w2[E,H,I], ids[T,R]")
        if scores.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected FP32 top-k scores and INT32/INT64 expert ids")
        if scores.shape != ids.shape or ids.shape[0] != x.shape[0]:
            raise ValueError("Routing tensors must have identical shape [T,R]")
        if w1.shape != (w2.shape[0], 2 * w2.shape[2], x.shape[1]) or w2.shape[1] != x.shape[1]:
            raise ValueError("Incompatible expert weight dimensions")
        if not 1 <= ids.shape[1] <= w1.shape[0]:
            raise ValueError("Require 1 <= top-k <= E")
        if any(t.device != x.device for t in (w1, w2, scores, ids)) or x.device.type != "cuda":
            raise ValueError("All inputs must be on the same CUDA device")
        if not x.is_contiguous() or w1.stride(-1) != 1 or w2.stride(-1) != 1 or scores.stride(-1) != 1:
            raise ValueError("Use contiguous activations/scores and K-contiguous Linear weights")
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, w1, w2, scores)):
            raise ValueError("The original vLLM functional FFN is inference only")
        return self.module.fused_experts_impl(x, w1, w2, scores, ids)


def load_vllm_experts(vllm_dir: str | Path, *, arch="11.0a", verbose=False):
    """Return ``runner(x,w1_linear,w2_linear,scores_fp32,ids), metadata``.

    The runner exposes ``config_for(T,w1,w2,R)`` for saving the actual upstream
    tile choice before timing. Native compilation/metadata setup are outside
    forward timing. Only original default single-GPU BF16/SiLU is exposed.
    """
    from vllm_moe_native import load_native_ops

    root = Path(vllm_dir).expanduser().resolve()
    metadata = _source_metadata(root)
    native_ops, native_metadata = load_native_ops(root, arch=arch, verbose=verbose)
    device_name = torch.cuda.get_device_name()
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (11, 0):
        raise NotImplementedError("This pinned comparison is validated for Thor SM110 only")
    module, envs = _load_core(root, native_ops, device_name=device_name, capability=capability)
    metadata.update(
        loader="unchanged upstream AST definitions plus original upstream CUDA ops",
        upstream_entrypoint="fused_experts_impl", backend="Triton BF16 functional",
        cuda_graph_capture=False, native=native_metadata, configurations={},
        device_name=device_name, compute_capability=list(capability),
        extracted_core_definitions=list(_CORE_NAMES),
        shims=[
            "Calls unchanged fused_experts_impl directly; excludes the serving-facing torch.ops.vllm.fused_experts custom-op dispatch wrapper",
            "No serving-package import: private module import hook supplies default get_config()=None and CUDA platform queries",
            "Private torch proxy maps the unchanged activation's torch.ops._C.silu_and_mul to the original CUDA dispatcher binding",
            "Standard-library re supplies upstream device-name normalization's identical whitespace/slash expression",
            "Unquantized moe_kernel_quantize_input takes its original no-op branch; quantized/EP/nonstandard activation paths are not exposed",
            "No CUDA tensor descriptors: upstream default VLLM_TRITON_USE_TD unset/0; no runtime tile override",
        ],
        environment=dict(VLLM_TRITON_USE_TD=os.environ.get("VLLM_TRITON_USE_TD"),
                         VLLM_BATCH_INVARIANT=False,
                         VLLM_TUNED_CONFIG_FOLDER=envs.VLLM_TUNED_CONFIG_FOLDER),
        weights="Original Linear format [E,2I,H] and [E,H,I], last dimension contiguous; no timed copy",
        score_position="FP32 down accumulator times routing score, then BF16 cast; top-k summed in FP32",
        scope="Original expert alignment/counts, two Triton GEMMs, original CUDA SiLU/multiply and top-k sum; excludes router/top-k selection",
    )
    return _Runner(module, root, envs, metadata), metadata
