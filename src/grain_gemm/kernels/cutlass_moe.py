# SPDX-License-Identifier: Apache-2.0
"""Independent Thor MoE-only CUTLASS tile families; single GEMM stays separate."""

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).parent
_LIBRARY = _ROOT / "_native/libgrain_cutlass_moe.so"
CONFIGS = {
    config: dict(block_m=m, block_n=n, block_k=256, cluster_m=cluster,
                 scheduler="static" if static else "dynamic", counts_aware=config in (43, 44))
    for config, m, n, cluster, static in (
        (34, 128, 256, 2, False), (35, 128, 128, 2, False), (36, 256, 128, 2, False),
        (37, 128, 256, 1, False), (38, 128, 128, 1, False),
        (41, 128, 256, 2, True), (42, 128, 128, 1, True),
        (43, 128, 256, 2, True), (44, 128, 128, 1, True),
    )
}


def source_hashes():
    target = _ROOT / "csrc/sm110"
    sources = (target / "int8_g256_moe.cu", *(p for p in sorted((target / "include").rglob("*"))
                                                if p.suffix in (".h", ".hpp")))
    return {str(p.relative_to(_ROOT / "csrc")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources}


@lru_cache(maxsize=1)
def _load_library():
    if not _LIBRARY.is_file():
        raise RuntimeError("Build the Thor MoE library with tools/build_cutlass_moe.py --cutlass-dir PATH")
    try:
        metadata = json.loads(_LIBRARY.with_name("moe_cutlass_build.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Thor MoE build metadata is missing or invalid; rebuild") from exc
    if metadata.get("architecture") != "sm_110" or metadata.get("source_sha256") != source_hashes():
        raise RuntimeError("Thor MoE library is stale or targets another architecture; rebuild")
    library = ctypes.CDLL(str(_LIBRARY))
    library.grain_cutlass_moe_workspace_bytes.argtypes = [ctypes.c_int] * 7
    library.grain_cutlass_moe_workspace_bytes.restype = ctypes.c_size_t
    library.grain_cutlass_moe_g256.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 7
                                            + [ctypes.c_void_p] * 3 + [ctypes.c_size_t])
    library.grain_cutlass_moe_g256.restype = ctypes.c_int
    library.grain_cutlass_moe_error_string.argtypes = [ctypes.c_int]
    library.grain_cutlass_moe_error_string.restype = ctypes.c_char_p
    return library


def is_available(device=None):
    import torch
    if torch.cuda.get_device_capability(device) != (11, 0):
        return False
    try:
        _load_library()
    except (OSError, RuntimeError, AttributeError):
        return False
    return True


def launch(a, weight, scale_a, scale_b, out, config_id=34, *, counts=None, swizzle=1, raster=0):
    """G256 batched GEMM using contiguous A[E,M,K], B[E,N,K] and BF16 D.

    Scale tensors retain public layouts SA[E,M,K/256], SB[E,K/256,N].
    Raster 0 is heuristic, 1 along N, 2 along M; swizzle is 1/2/4/8.
    Counts-aware configs accept GPU int32 counts[E] in [0,M]; only each
    expert's live row prefix is defined. None computes all M rows. Other
    configs ignore counts. No values are synchronized to the host.
    """
    import torch
    if type(config_id) is not int or config_id not in CONFIGS:
        raise ValueError(f"Thor MoE config_id must be one of {tuple(CONFIGS)}")
    if type(swizzle) is not int or swizzle not in (1, 2, 4, 8):
        raise ValueError("swizzle must be 1, 2, 4 or 8")
    if type(raster) is not int or raster not in (0, 1, 2):
        raise ValueError("raster must be 0 (heuristic), 1 (N) or 2 (M)")
    tensors = (("a", a, torch.int8), ("weight", weight, torch.int8),
               ("scale_a", scale_a, torch.float32), ("scale_b", scale_b, torch.float32),
               ("out", out, torch.bfloat16))
    for name, tensor, dtype in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.ndim != 3:
            raise ValueError(f"{name} must be a three-dimensional strided Tensor")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tensor.device.type != "cuda" or tensor.device != a.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the same CUDA device as a")
        if tensor.numel() > 2**31 or any(d > 2**31 - 1 for d in tensor.shape):
            raise ValueError(f"{name} exceeds the supported indexing range")
        if tensor.data_ptr() % (4 if name == "scale_a" else 16):
            raise ValueError(f"{name} is not aligned for the Thor MoE collective")
    e, m, k = a.shape
    n = weight.shape[1]
    if min(e, m, n, k) <= 0 or m % 64 or n % 64 or k % 256:
        raise ValueError("Thor MoE requires positive E/M/N/K, M/N multiples of 64 and K multiple of 256")
    for name, tensor, shape in (("weight", weight, (e,n,k)), ("scale_a", scale_a, (e,m,k//256)),
                                ("scale_b", scale_b, (e,k//256,n)), ("out", out, (e,m,n))):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if torch.cuda.get_device_capability(a.device) != (11, 0):
        raise ValueError("This MoE collective requires SM110")
    if counts is not None:
        if not isinstance(counts, torch.Tensor) or counts.layout != torch.strided or counts.ndim != 1:
            raise ValueError("counts must be a one-dimensional strided Tensor")
        if counts.dtype != torch.int32:
            raise TypeError("counts must have dtype torch.int32")
        if counts.device != a.device or not counts.is_contiguous() or tuple(counts.shape) != (e,):
            raise ValueError("counts must be contiguous with shape [E] on the same CUDA device as a")
        if counts.data_ptr() % 4:
            raise ValueError("counts must have 4-byte alignment")
    with torch.cuda.device(a.device):
        return _launch_prevalidated(a, weight, scale_a, scale_b, out, config_id,
                                   counts=counts, swizzle=swizzle, raster=raster)


def _launch_prevalidated(a, weight, scale_a, scale_b, out, config_id=34, *,
                         counts=None, swizzle=1, raster=0):
    """Internal launch after caller validation and CUDA device selection.

    Workspace belongs to this call, including under graph capture. Tensor
    pointers and kernel arguments are never cached across calls.
    """
    import torch
    e, m, k = a.shape
    n = weight.shape[1]
    library = _load_library()
    stream = torch.cuda.current_stream(a.device).cuda_stream
    size = library.grain_cutlass_moe_workspace_bytes(m,n,k,e,config_id,swizzle,raster)
    workspace = torch.empty(size, dtype=torch.uint8, device=a.device) if size else None
    status = library.grain_cutlass_moe_g256(
        a.data_ptr(), weight.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(),
        m,n,k,e,config_id,swizzle,raster,counts.data_ptr() if counts is not None else 0,
        stream,workspace.data_ptr() if workspace is not None else 0,size)
    if status:
        message = library.grain_cutlass_moe_error_string(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"Thor MoE CUTLASS launch failed ({status}): {message}")
    return out
