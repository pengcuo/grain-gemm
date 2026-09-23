"""Optional native CuTe INT8 G256 kernels, built explicitly with tools/build_cuda.py."""

import ctypes
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ._native import SUPPORTED_ARCHITECTURES, architecture_for_device

_LIBRARY = Path(__file__).parent / "_native" / "libgrain_cuda.so"
_TILES = (
    (64, 64, 128, 2, 4),
    (64, 64, 128, 3, 4),
    (128, 64, 128, 2, 4),
    (128, 64, 128, 3, 4),
    (64, 128, 128, 2, 4),
    (64, 128, 128, 3, 4),
    (128, 128, 128, 2, 8),
    (128, 128, 128, 3, 8),
)
# IDs 0–7 use register-prefetched scales and a shared-memory output transpose;
# 8–15 stage scales asynchronously and write adjacent BF16 pairs directly.
CONFIGS = tuple(
    dict(block_m=m, block_n=n, block_k=k, num_stages=stages,
         num_warps=warps, scale_load=scale_load, store_bits=store_bits)
    for scale_load, store_bits in (("register_prefetch", 128), ("async_shared", 32))
    for m, n, k, stages, warps in _TILES
) + (
    dict(block_m=128, block_n=128, block_k=128, num_stages=3,
         num_warps=8, scale_load="async_shared", store_bits=16),
)


@lru_cache(maxsize=2)
def _load_library(architecture):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"Unsupported native target: {architecture}")
    if not _LIBRARY.is_file():
        raise RuntimeError(
            "The optional GrainGEMM CUDA library has not been built. From the "
            "repository root, run: python tools/build_cuda.py "
            f"--cutlass-dir /path/to/cutlass --arch {architecture}"
        )
    manifest = _LIBRARY.with_name("build.json")
    source = Path(__file__).parent / "csrc" / "sm12x" / "int8_g256_cute.cu"
    try:
        info = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Native build metadata is missing or invalid; rebuild with tools/build_cuda.py") from exc
    if info.get("architecture") != architecture:
        raise RuntimeError(
            f"Native library targets {info.get('architecture')}, but the device requires "
            f"{architecture}; rebuild with tools/build_cuda.py --arch {architecture}"
        )
    if info.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Native library is stale; rebuild with tools/build_cuda.py")
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError as exc:
        raise RuntimeError(f"Cannot load GrainGEMM CUDA library {_LIBRARY}: {exc}") from exc
    library.grain_cute_g256.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    library.grain_cute_g256.restype = ctypes.c_int
    library.grain_cuda_error_string.argtypes = [ctypes.c_int]
    library.grain_cuda_error_string.restype = ctypes.c_char_p
    return library


def is_available(device=None):
    """Whether a native build matches the source and the requested CUDA device."""
    try:
        _load_library(architecture_for_device(device))
    except (RuntimeError, AttributeError, AssertionError):
        return False
    return True


def launch(a, b, scale_a, scale_b, out, config_id=0):
    """Launch on the current PyTorch CUDA stream; the public API validates tensors.

    B is logical [K, N] with contiguous [N, K] backing storage; scale_b is
    contiguous [K / 256, N]. No transpose, packing, or allocation occurs here.
    """
    import torch

    if isinstance(config_id, bool) or not isinstance(config_id, int) or not 0 <= config_id < len(CONFIGS):
        raise ValueError(f"config_id must be an integer in [0, {len(CONFIGS) - 1}]")
    library = _load_library(architecture_for_device(a.device))
    m, k = a.shape
    n = b.shape[1]
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(a.device).cuda_stream
        status = library.grain_cute_g256(
            a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
            out.data_ptr(), m, n, k, config_id, stream,
        )
    if status:
        message = library.grain_cuda_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"GrainGEMM CUDA launch failed ({status}): {message}")
    return out
