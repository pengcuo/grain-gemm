# SPDX-License-Identifier: Apache-2.0
"""Optional CUTLASS GemmUniversal backend with a custom INT8 G256 collective."""

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).parent
_LIBRARY = _ROOT / "_native" / "libgrain_cutlass.so"
CONFIGS = tuple(
    dict(block_m=m, block_n=n, block_k=128, num_stages=stages, num_warps=warps)
    for m, n, stages, warps in (
        (64, 64, 2, 4), (64, 64, 3, 4),
        (128, 64, 2, 4), (128, 64, 3, 4),
        (64, 128, 2, 4), (64, 128, 3, 4),
        (128, 128, 2, 8), (128, 128, 3, 8),
    )
)


def _source_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((_ROOT / "csrc").glob("grain_cutlass*"))
        if path.suffix in (".cu", ".hpp")
    }


@lru_cache(maxsize=1)
def _load_library():
    if not _LIBRARY.is_file():
        raise RuntimeError(
            "The optional CUTLASS library has not been built; run "
            "python tools/build_cutlass.py --cutlass-dir /path/to/cutlass"
        )
    try:
        metadata = json.loads(_LIBRARY.with_name("cutlass_build.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("CUTLASS build metadata is missing or invalid; rebuild") from exc
    if (metadata.get("architecture") != "sm_121"
            or metadata.get("source_sha256") != _source_hashes()):
        raise RuntimeError("CUTLASS library is stale or targets a different architecture; rebuild for sm_121")
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError as exc:
        raise RuntimeError(f"Cannot load CUTLASS library {_LIBRARY}: {exc}") from exc
    library.grain_cutlass_g256.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    library.grain_cutlass_g256.restype = ctypes.c_int
    library.grain_cutlass_error_string.argtypes = [ctypes.c_int]
    library.grain_cutlass_error_string.restype = ctypes.c_char_p
    return library


def is_available():
    """Whether a native library matching the local source can be loaded."""
    try:
        _load_library()
    except (RuntimeError, AttributeError):
        return False
    return True


def launch(a, b, scale_a, scale_b, out, config_id=0):
    """Launch on PyTorch's current stream; the public API validates the layout."""
    import torch

    if isinstance(config_id, bool) or not isinstance(config_id, int) or not 0 <= config_id < len(CONFIGS):
        raise ValueError(f"config_id must be an integer in [0, {len(CONFIGS) - 1}]")
    library = _load_library()
    m, k = a.shape
    n = b.shape[1]
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(a.device).cuda_stream
        status = library.grain_cutlass_g256(
            a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
            out.data_ptr(), m, n, k, config_id, stream,
        )
    if status:
        message = library.grain_cutlass_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"GrainGEMM CUTLASS launch failed ({status}): {message}")
    return out
