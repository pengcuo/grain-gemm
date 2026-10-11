# SPDX-License-Identifier: Apache-2.0
"""Optional CUTLASS GemmUniversal backend with a custom INT8 G256 collective."""

import ctypes
from functools import lru_cache
import json
from pathlib import Path

from ._native import SUPPORTED_ARCHITECTURES, architecture_for_device, cutlass_source_hash

_ROOT = Path(__file__).parent
_LIBRARY = _ROOT / "_native" / "libgrain_cutlass.so"
_TILES = (
        (64, 64, 2, 4), (64, 64, 3, 4),
        (128, 64, 2, 4), (128, 64, 3, 4),
        (64, 128, 2, 4), (64, 128, 3, 4),
        (128, 128, 2, 8), (128, 128, 3, 8),
)
# Keep IDs 0–7 stable for existing measured dispatch entries. The additional
# families allow the output exchange and asynchronous scales to be compared
# independently with identical MMA tiles and pipeline depths.
CONFIGS = tuple(
    dict(block_m=m, block_n=n, block_k=128, num_stages=stages,
         num_warps=warps, scale_load=scale_load, epilogue=epilogue)
    for scale_load, epilogue in (
        ("register_prefetch", "fp32_shared"),
        ("register_prefetch", "bf16_shared"),
        ("async_shared", "bf16_direct"),
        ("register_prefetch", "bf16_direct"),
    )
    for m, n, stages, warps in _TILES
)
SM110_CONFIGS = CONFIGS + tuple(
    dict(block_m=256, block_n=256, block_k=256, num_stages="auto",
         num_warps=12, cluster_m=2, cluster_n=1, scale_load="async_shared",
         epilogue=f"tma_bf16_128x{epilogue_n}", operand_load="tma",
         promotion_warps=8, scale_a_layout="K_major", scale_b_layout="MN_major")
    for epilogue_n in (32, 16)
)


def configs_for_device(device=None):
    return SM110_CONFIGS if architecture_for_device(device) == "sm_110" else CONFIGS


def _source_hashes(architecture="sm_121"):
    return cutlass_source_hash(architecture)


@lru_cache(maxsize=3)
def _load_library(architecture):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"Unsupported native target: {architecture}")
    if not _LIBRARY.is_file():
        raise RuntimeError(
            "The optional CUTLASS library has not been built; run "
            f"python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch {architecture}"
        )
    try:
        metadata = json.loads(_LIBRARY.with_name("cutlass_build.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("CUTLASS build metadata is missing or invalid; rebuild") from exc
    if metadata.get("architecture") != architecture:
        raise RuntimeError(
            f"CUTLASS library targets {metadata.get('architecture')}, but the device requires "
            f"{architecture}; rebuild with tools/build_cutlass.py --arch {architecture}"
        )
    if metadata.get("source_sha256") != _source_hashes(architecture):
        raise RuntimeError("CUTLASS library is stale; rebuild with tools/build_cutlass.py")
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError as exc:
        raise RuntimeError(f"Cannot load CUTLASS library {_LIBRARY}: {exc}") from exc
    library.grain_cutlass_g256.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    library.grain_cutlass_g256.restype = ctypes.c_int
    library.grain_cutlass_error_string.argtypes = [ctypes.c_int]
    library.grain_cutlass_error_string.restype = ctypes.c_char_p
    if architecture == "sm_110":
        library.grain_cutlass_workspace_bytes.argtypes = [ctypes.c_int] * 4
        library.grain_cutlass_workspace_bytes.restype = ctypes.c_size_t
        library.grain_cutlass_g256_sm110.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 +
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        )
        library.grain_cutlass_g256_sm110.restype = ctypes.c_int
    return library


def is_available(device=None):
    """Whether a native build matches the source and the requested CUDA device."""
    try:
        _load_library(architecture_for_device(device))
    except (RuntimeError, AttributeError, AssertionError):
        return False
    return True


def launch(a, b, scale_a, scale_b, out, config_id=0):
    """Launch on PyTorch's current stream; the public API validates the layout."""
    import torch

    architecture = architecture_for_device(a.device)
    configs = SM110_CONFIGS if architecture == "sm_110" else CONFIGS
    if isinstance(config_id, bool) or not isinstance(config_id, int) or not 0 <= config_id < len(configs):
        raise ValueError(f"config_id must be an integer in [0, {len(configs) - 1}]")
    library = _load_library(architecture)
    m, k = a.shape
    n = b.shape[1]
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(a.device).cuda_stream
        args = (a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
                out.data_ptr(), m, n, k, config_id, stream)
        if architecture == "sm_110" and config_id >= len(CONFIGS):
            size = library.grain_cutlass_workspace_bytes(m, n, k, config_id)
            # PyTorch owns this stream-local allocation, including during graph
            # capture. Reinitialize scheduler state for each launch; no handles
            # or pointer-dependent parameters are cached across calls/streams.
            workspace = torch.empty(size, device=a.device, dtype=torch.uint8) if size else None
            status = library.grain_cutlass_g256_sm110(
                *args, workspace.data_ptr() if workspace is not None else 0, size,
            )
        else:
            status = library.grain_cutlass_g256(*args)
    if status:
        message = library.grain_cutlass_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"GrainGEMM CUTLASS launch failed ({status}): {message}")
    return out
