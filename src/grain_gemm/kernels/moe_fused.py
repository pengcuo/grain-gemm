# SPDX-License-Identifier: Apache-2.0
"""Optional SM110 fused gate/up, SwiGLU and G256 activation quantization.

This private bridge consumes the existing expert-major packed input and
Linear weights. It never materializes a BF16 gate/up output or repacks weights.
The public MoE dispatcher decides whether a measured shape should use it.
"""

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).parent
_LIBRARY = _ROOT / "_native/libgrain_moe_fused.so"
_METADATA_NAME = "moe_fused_build.json"
_ABI_VERSION = 1
CONSUMER_WARPS = (8, 16)


def source_hashes():
    """Hashes of this library's packaged CUDA translation units."""
    source = _ROOT / "csrc/sm110/int8_g256_moe_fused.cu"
    return {str(source.relative_to(_ROOT / "csrc")):
            hashlib.sha256(source.read_bytes()).hexdigest()}


@lru_cache(maxsize=1)
def _load_library():
    if not _LIBRARY.is_file():
        raise RuntimeError("Build the optional Thor fused MoE library with "
                           "tools/build_moe_fused.py --cutlass-dir PATH")
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
    library.grain_moe_fused_launch.argtypes = ([ctypes.c_void_p] * 7
                                             + [ctypes.c_int] * 6 + [ctypes.c_void_p])
    library.grain_moe_fused_launch.restype = ctypes.c_int
    library.grain_moe_fused_smem_bytes.argtypes = [ctypes.c_int]
    library.grain_moe_fused_smem_bytes.restype = ctypes.c_size_t
    library.grain_moe_fused_error_string.argtypes = [ctypes.c_int]
    library.grain_moe_fused_error_string.restype = ctypes.c_char_p
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
    return _load_library().grain_moe_fused_smem_bytes(consumer_warps)


def _launch_prevalidated(a, weight, scale_a, scale_b, counts, out, scale_out, *,
                         consumer_warps=16, persistent_blocks=0):
    """Launch after public validation, SM110 selection and CUDA device selection.

    A[E,C,H] and Linear weight[E,2I,H] are contiguous INT8. FP32 scales
    retain SA[E,C,H/256] and SB[E,H/256,2I]. Counts[E] is GPU INT32;
    outputs are contiguous INT8 Q[E,C,I] and FP32 S[E,C,I/256]. C is a
    multiple of 64; H and I are multiples of 256. Counts lie in [0,C].
    Padding is defined as Q=0 and S=1e-12/127, matching swiglu_quantize.

    No counts or tensor values are copied to the CPU. There is no mutable
    workspace or pointer cache: independent calls own their output buffers,
    including across streams and graph captures. persistent_blocks specifies
    physical CTAs, rounded up to a multiple of two for two-SM clusters. Zero
    starts from the device's SM count; the cluster count is capped by available
    cluster tasks.
    """
    import torch
    if type(consumer_warps) is not int or consumer_warps not in CONSUMER_WARPS:
        raise ValueError(f"consumer_warps must be one of {CONSUMER_WARPS}")
    if type(persistent_blocks) is not int or not 0 <= persistent_blocks <= 2**31 - 1:
        raise ValueError("persistent_blocks must be a nonnegative signed 32-bit integer")
    e, c, h = a.shape
    i = weight.shape[1] // 2
    library = _load_library()
    stream = torch.cuda.current_stream(a.device).cuda_stream
    status = library.grain_moe_fused_launch(
        a.data_ptr(), weight.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
        counts.data_ptr(), out.data_ptr(), scale_out.data_ptr(),
        c, i, h, e, consumer_warps, persistent_blocks, stream)
    if status:
        message = library.grain_moe_fused_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"Thor fused MoE launch failed ({status}): {message}")
    return out, scale_out
