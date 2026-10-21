# SPDX-License-Identifier: Apache-2.0
"""SM110 original-token gate/up, SwiGLU and G256 quantization.

Reads INT8 x[T,H] and FP32 scale_x[T,H/256] directly using stable expert
route indices. No expert-expanded global input or input-scale buffer exists.
"""

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).parent
_LIBRARY = _ROOT / "_native/libgrain_moe_gather.so"
_METADATA_NAME = "moe_gather_build.json"
_ABI_VERSION = 1
CONSUMER_WARPS = (8, 16)


def source_hashes():
    """Hashes of this library's packaged CUDA translation units."""
    source = _ROOT / "csrc/sm110/int8_g256_moe_gather.cu"
    return {str(source.relative_to(_ROOT / "csrc")):
            hashlib.sha256(source.read_bytes()).hexdigest()}


@lru_cache(maxsize=1)
def _load_library():
    if not _LIBRARY.is_file():
        raise RuntimeError("Build the optional Thor fused MoE library with "
                           "tools/build_moe_gather.py --cutlass-dir PATH")
    try:
        metadata = json.loads(_LIBRARY.with_name(_METADATA_NAME).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Thor fused MoE build metadata is missing or invalid; rebuild") from exc
    if (not isinstance(metadata, dict)
            or metadata.get("architecture") != "sm_110"
            or metadata.get("compiler_architecture") != "sm_110a"
            or metadata.get("abi_version") != _ABI_VERSION
            or metadata.get("consumer_warps") != list(CONSUMER_WARPS)
            or metadata.get("source_sha256") != source_hashes()):
        raise RuntimeError("Thor fused MoE library is stale or incompatible; rebuild")
    if metadata.get("library_sha256") != hashlib.sha256(_LIBRARY.read_bytes()).hexdigest():
        raise RuntimeError("Thor fused MoE library does not match its build metadata; rebuild")
    library = ctypes.CDLL(str(_LIBRARY))
    library.grain_moe_gather_launch.argtypes = ([ctypes.c_void_p] * 9
                                              + [ctypes.c_int] * 8 + [ctypes.c_void_p])
    library.grain_moe_gather_launch.restype = ctypes.c_int
    library.grain_moe_gather_smem_bytes.argtypes = [ctypes.c_int]
    library.grain_moe_gather_smem_bytes.restype = ctypes.c_size_t
    library.grain_moe_gather_error_string.argtypes = [ctypes.c_int]
    library.grain_moe_gather_error_string.restype = ctypes.c_char_p
    return library


def is_available(device=None):
    """Return False for other architectures and absent/stale optional builds."""
    import torch
    if torch.cuda.get_device_capability(device) != (11, 0):
        return False
    try:
        _load_library()
    except (OSError, RuntimeError, AttributeError):
        return False
    return True


def shared_memory_bytes(consumer_warps=16):
    if type(consumer_warps) is not int or consumer_warps not in CONSUMER_WARPS:
        raise ValueError(f"consumer_warps must be one of {CONSUMER_WARPS}")
    return _load_library().grain_moe_gather_smem_bytes(consumer_warps)


def _launch_prevalidated(x, weight, scale_x, scale_b, counts, offsets, order,
                         out, scale_out, *, top_k, consumer_warps=16,
                         persistent_blocks=0):
    """Read original x/scales and route indices; write only QH and SH.

    Public validation establishes contiguous x[T,H] INT8 and scale_x[T,H/256]
    FP32, Linear weight[E,2I,H], FP32 scale_b[E,H/256,2I], GPU INT32 counts[E]
    and offsets[E+1], and INT64 order[T*R]. x and weights require 16-byte
    alignment. Output QH[E,C,I] INT8 and SH[E,C,I/256] FP32 have C%64==0.
    Masked/unused token rows are not loaded. Padding outputs are initialized
    to Q=0 and scale=1e-12/127, matching the existing SwiGLU quantizer.
    All metadata remains on the current stream and is reread on graph replay.
    """
    import torch
    if type(consumer_warps) is not int or consumer_warps not in CONSUMER_WARPS:
        raise ValueError(f"consumer_warps must be one of {CONSUMER_WARPS}")
    if type(persistent_blocks) is not int or not 0 <= persistent_blocks <= 2**31 - 1:
        raise ValueError("persistent_blocks must be a nonnegative signed 32-bit integer")
    e, twice_i, h = weight.shape
    c = out.shape[1]
    library = _load_library()
    status = library.grain_moe_gather_launch(
        x.data_ptr(), scale_x.data_ptr(), weight.data_ptr(), scale_b.data_ptr(),
        counts.data_ptr(), offsets.data_ptr(), order.data_ptr(),
        out.data_ptr(), scale_out.data_ptr(), c, twice_i // 2, h, e,
        x.shape[0], top_k, consumer_warps, persistent_blocks,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if status:
        message = library.grain_moe_gather_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"Thor original-token MoE launch failed ({status}): {message}")
    return out, scale_out
