"""Explicit experimental Thor Mega FFN. Importing does not initialize CUDA.

This module is separate from grain_gemm.moe_ffn and never changes auto dispatch.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading

import torch

HERE=Path(__file__).resolve().parent
DEFAULT_LIBRARY=HERE/'build/libgrain_moe_mega.so'
CUTLASS_COMMIT='098de2a652cf8f00fd70b2df54051c7eccbb855a'
_INITIALIZATION_LOCK=threading.Lock()
_INITIALIZED={}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {path.name:_sha(path) for path in sorted(HERE.iterdir())
            if path.suffix in ('.cu','.cuh','.hpp','.h')}


@lru_cache(maxsize=None)
def _load_library(path):
    path=Path(path).expanduser().resolve()
    try:
        metadata=json.loads(path.with_suffix('.json').read_text())
    except (OSError,ValueError) as error:
        raise RuntimeError('Build the explicit Mega experiment with its build.py') from error
    if (not isinstance(metadata,dict) or metadata.get('abi_version')!=1
            or metadata.get('architecture')!='sm_110'
            or metadata.get('compiler_architecture')!='sm_110a'
            or metadata.get('consumer_warps')!=[8,16]
            or metadata.get('cutlass_commit')!=CUTLASS_COMMIT
            or type(metadata.get('silu_lut')) is not bool
            or metadata.get('source_sha256')!=_source_hashes()
            or metadata.get('library_sha256')!=_sha(path)):
        raise RuntimeError('Experimental Mega build is stale or incompatible; rebuild')
    library=ctypes.CDLL(str(path))
    library.grain_moe_mega_cluster_launch.argtypes=([ctypes.c_void_p]*12+[ctypes.c_int]*8+[ctypes.c_void_p])
    library.grain_moe_mega_cluster_launch.restype=ctypes.c_int
    library.grain_moe_mega_cluster_error_string.argtypes=[ctypes.c_int]
    library.grain_moe_mega_cluster_error_string.restype=ctypes.c_char_p
    library.grain_moe_mega_cluster_smem_bytes.argtypes=[ctypes.c_int]
    library.grain_moe_mega_cluster_smem_bytes.restype=ctypes.c_size_t
    initializer=getattr(library,'grain_moe_mega_cluster_initialize',None)
    if bool(initializer)!=metadata['silu_lut']:
        raise RuntimeError('Mega build metadata disagrees with its SiLU initialization ABI')
    if initializer is not None:
        initializer.argtypes=[ctypes.c_void_p]
        initializer.restype=ctypes.c_int
    return library,metadata


def build_metadata(library=None):
    """Return first-load-checked metadata; restart after rebuilding a loaded path."""
    path=Path(library or DEFAULT_LIBRARY).expanduser().resolve()
    _,metadata=_load_library(str(path))
    return dict(metadata,library=str(path),source=str(HERE/'mega_cluster.cu'))


def _initialize_optional(library,path,device,stream):
    """Initialize an optional immutable LUT once per library/device, safely.

    The default kernel has no initializer. The exact-LUT build may
    export initialize(stream), whose only action is an asynchronous fill. Its
    completion event makes another stream wait for the initialized data. The
    lock is also needed because ctypes releases the Python GIL during C calls.
    """
    initializer=getattr(library,'grain_moe_mega_cluster_initialize',None)
    if initializer is None:
        return
    key=(str(path),device.index)
    with _INITIALIZATION_LOCK:
        ready=_INITIALIZED.get(key)
        if ready is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Warm up the Mega experiment before capture')
            status=initializer(stream.cuda_stream)
            if status:
                raise RuntimeError('Mega optional initialization failed: '+str(status))
            event=torch.cuda.Event()
            event.record(stream)
            _INITIALIZED[key]=(event,stream.cuda_stream)
        elif ready[1]!=stream.cuda_stream:
            stream.wait_event(ready[0])


def _run(hidden_states,w_gate_up,w_down,topk_weights,topk_ids,*,
         scale_x,scale_gate_up,scale_down,group_size=256,activation='swiglu',
         token_mask=None,output_dtype=torch.bfloat16,backend='auto',
         gate_up_layout='interleaved',consumer_warps=16,persistent_blocks=0,
         library=None,debug=False):
    from grain_gemm.moe import _validate
    from grain_gemm.kernels import moe as kernels
    if gate_up_layout!='interleaved':
        raise ValueError("moe_ffn_mega requires gate_up_layout='interleaved'")
    if backend not in ('auto','cutlass'):
        raise ValueError("moe_ffn_mega requires backend='auto' or 'cutlass'")
    if output_dtype!=torch.bfloat16:
        raise ValueError('moe_ffn_mega currently requires BF16 output')
    if type(consumer_warps) is not int or consumer_warps not in (8,16):
        raise ValueError('consumer_warps must be 8 or 16')
    if type(persistent_blocks) is not int or not 0<=persistent_blocks<=2**31-1:
        raise ValueError('persistent_blocks must be a nonnegative signed 32-bit integer')
    tokens,hidden,intermediate,experts,top_k=_validate(
        hidden_states,w_gate_up,w_down,topk_weights,topk_ids,scale_x,scale_gate_up,
        scale_down,token_mask,group_size,activation,backend,output_dtype,gate_up_layout)
    if (hidden,intermediate)!=(1536,1024):
        raise ValueError('This Mega prototype supports only H=1536 and I=1024')
    if torch.cuda.get_device_capability(hidden_states.device)!=(11,0):
        raise RuntimeError('This Mega prototype requires Thor SM110')
    with torch.cuda.device(hidden_states.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('This experimental Mega path requires eager capacity statistics')
        if not tokens:
            output=torch.zeros(hidden_states.shape,device=hidden_states.device,dtype=output_dtype)
            if not debug:
                return output
            counts=torch.zeros(experts,device=hidden_states.device,dtype=torch.int32)
            return dict(output=output,qh=torch.empty((experts,0,intermediate),device=hidden_states.device,dtype=torch.int8),
                sh=torch.empty((experts,0,intermediate//256),device=hidden_states.device,dtype=torch.float32),
                expert_y=torch.empty((experts,0,hidden),device=hidden_states.device,dtype=output_dtype),counts=counts,capacity=0)
        routes=kernels.prepare_routes(topk_ids,experts,token_mask,include_tiles=False)
        invalid,maximum,total=routes.statistics.cpu().tolist()
        if invalid:
            raise ValueError('unmasked topk_ids must be in [0, E)')
        if not maximum:
            output=torch.zeros(hidden_states.shape,device=hidden_states.device,dtype=output_dtype)
            if not debug:
                return output
            return dict(output=output,qh=torch.empty((experts,0,intermediate),device=hidden_states.device,dtype=torch.int8),
                sh=torch.empty((experts,0,intermediate//256),device=hidden_states.device,dtype=torch.float32),
                expert_y=torch.empty((experts,0,hidden),device=hidden_states.device,dtype=output_dtype),counts=routes.counts,capacity=0)
        if any(value.data_ptr()%16 for value in (hidden_states,scale_x,w_gate_up,w_down,scale_gate_up,scale_down)):
            raise ValueError('Experimental Mega inputs/weights/scales require 16-byte alignment')
        capacity=((maximum+63)//64)*64
        if max(experts*capacity*hidden,experts*capacity*2*intermediate)>2**31-1:
            raise ValueError('native MoE workspace exceeds the supported indexing range')
        path=Path(library or DEFAULT_LIBRARY).expanduser().resolve()
        native,_=_load_library(str(path))
        stream=torch.cuda.current_stream(hidden_states.device)
        _initialize_optional(native,path,hidden_states.device,stream)
        expert_y=torch.empty((experts,capacity,hidden),device=hidden_states.device,dtype=output_dtype)
        qh=torch.empty((experts,capacity,intermediate),device=hidden_states.device,dtype=torch.int8) if debug else None
        sh=torch.empty((experts,capacity,intermediate//256),device=hidden_states.device,dtype=torch.float32) if debug else None
        status=native.grain_moe_mega_cluster_launch(hidden_states.data_ptr(),scale_x.data_ptr(),
            w_gate_up.transpose(-2,-1).data_ptr(),scale_gate_up.data_ptr(),
            w_down.transpose(-2,-1).data_ptr(),scale_down.data_ptr(),routes.counts.data_ptr(),
            routes.offsets.data_ptr(),routes.order.data_ptr(),expert_y.data_ptr(),
            qh.data_ptr() if qh is not None else None,sh.data_ptr() if sh is not None else None,
            capacity,intermediate,hidden,experts,tokens,top_k,consumer_warps,persistent_blocks,stream.cuda_stream)
        if status:
            error=native.grain_moe_mega_cluster_error_string(status).decode('utf-8',errors='replace')
            raise RuntimeError(f'Experimental Mega launch failed ({status}): {error}')
        output=kernels.combine(expert_y,topk_weights,routes,output_dtype=output_dtype,packed_capacity=capacity)
        if debug:
            return dict(output=output,qh=qh,sh=sh,expert_y=expert_y,counts=routes.counts,capacity=capacity)
        return output


def moe_ffn_mega(hidden_states,w_gate_up,w_down,topk_weights,topk_ids,*,
                 scale_x,scale_gate_up,scale_down,group_size=256,activation='swiglu',
                 token_mask=None,output_dtype=torch.bfloat16,backend='auto',
                 gate_up_layout='interleaved',consumer_warps=16,persistent_blocks=0,
                 library=None):
    """Run the explicitly selected experimental interleaved SM110 Mega FFN.

    Public tensor/scale/routing conventions match grain_gemm.moe_ffn. The caller
    supplies interleaved [g0,u0,...] weight/scales, INT8 G256 original tokens and
    FP32 scales. This prototype requires H1536/I1024, BF16 output, SM110, eager
    execution and 16-byte-aligned pointers. Masked/empty rows return zero, empty
    experts and duplicate expert slots are supported, and scores are unchanged.
    Both projections preserve the original BF16 rounding. Quantized hidden and
    scales stay in cluster-local shared memory; expert outputs and combination
    remain global. No automatic dispatch, autograd or weight repacking occurs.
    """
    return _run(hidden_states,w_gate_up,w_down,topk_weights,topk_ids,
        scale_x=scale_x,scale_gate_up=scale_gate_up,scale_down=scale_down,
        group_size=group_size,activation=activation,token_mask=token_mask,
        output_dtype=output_dtype,backend=backend,gate_up_layout=gate_up_layout,
        consumer_warps=consumer_warps,persistent_blocks=persistent_blocks,library=library)


def debug_moe_ffn_mega(*args,**kwargs):
    """Untimed diagnostic launch returning real QH/SH/expert Y plus output."""
    return _run(*args,**kwargs,debug=True)
