"""Direct-output G32 FP8 input cast for the single-card MoE benchmark.

The finite BF16/FP16 input rule matches DeepGEMM's original
per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32,
                     use_packed_ue8m0=True).
Each UE8M0 exponent is ceil(log2(max(amax, 1e-4)/448)) with bias127;
four exponent bytes are packed into one int32. FP8 data use E4M3 RNE.
No routing, clock policy, weights or Mega kernel behavior are changed.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _direct_g32_fp8_input(X, Q, SF,
                         T: tl.constexpr, H: tl.constexpr,
                         X_ROW: tl.constexpr, X_COL: tl.constexpr,
                         Q_ROW: tl.constexpr, Q_COL: tl.constexpr,
                         SF_ROW: tl.constexpr, SF_COL: tl.constexpr,
                         ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    word = tl.program_id(1)
    cols = word * 128 + tl.arange(0, 128)
    values = tl.load(X + rows[:, None]*X_ROW + cols[None, :]*X_COL,
                     (rows[:, None] < T) & (cols[None, :] < H), other=0).to(tl.float32)
    groups = tl.reshape(values, (ROWS, 4, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), axis=2), 1.0e-4)
    # Use correctly rounded FP32 division, matching the original Torch helper
    # at exact exponent boundaries instead of relying on approximate division.
    raw_scale = libdevice.div_rn(maximum, 448.0)
    bits = raw_scale.to(tl.uint32, bitcast=True)
    exponents = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(tl.uint32)
    exponents = tl.minimum(tl.maximum(exponents, 1), 254)
    groups_in_word = word*4 + tl.arange(0, 4)
    # The original pack helper pads missing scale groups with1.0 (biased127).
    exponents = tl.where(groups_in_word[None, :] < tl.cdiv(H, 32), exponents, 127)
    inverse_bits = (254-exponents) << 23
    inverse = inverse_bits.to(tl.float32, bitcast=True)
    scaled = groups*inverse[:, :, None]
    quantized = tl.reshape(scaled, (ROWS, 128)).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Q + rows[:, None]*Q_ROW + cols[None, :]*Q_COL, quantized,
             (rows[:, None] < T) & (cols[None, :] < H))
    packed = tl.sum(exponents << (tl.arange(0, 4)[None, :]*8), axis=1).to(tl.int32)
    tl.store(SF + rows*SF_ROW + word*SF_COL, packed, rows < T)


def quantize_input_g32_into(x, out_q, out_sf, *, rows_per_program=8):
    """Write finite BF16/FP16 x[T,H] directly to supplied FP8/SF buffer views.

    `out_q` is E4M3 [capacity,H]; `out_sf` is int32
    [capacity,ceil(H/128)] containing packed UE8M0, usually row-major.
    Leading capacity may exceed T; arbitrary positive tensor strides work.
    Only x's T rows and H columns are written, so unused capacity is untouched.
    Returns None and performs no host/device synchronization or intermediate
    tensor allocation. The caller owns validation of finite input values.
    """
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("x must be a finite BF16/FP16 matrix")
    tokens, hidden = x.shape
    words = triton.cdiv(hidden, 128)
    if hidden <= 0:
        raise ValueError("hidden width must be positive")
    if out_q.dtype != torch.float8_e4m3fn or out_q.ndim != 2 or out_q.shape[0] < tokens or out_q.shape[1] != hidden:
        raise ValueError("out_q must be E4M3 [capacity,H], capacity>=T")
    if out_sf.dtype != torch.int32 or out_sf.ndim != 2 or out_sf.shape[0] < tokens or out_sf.shape[1] != words:
        raise ValueError("out_sf must be packed int32 [capacity,ceil(H/128)]")
    if x.device.type != "cuda" or x.device != out_q.device or x.device != out_sf.device:
        raise ValueError("x and both outputs must be on the same CUDA device")
    if any(stride <= 0 for tensor in (x,out_q,out_sf) for stride in tensor.stride()):
        raise ValueError("Only positive strides are supported")
    if rows_per_program not in (4,8,16):
        raise ValueError("rows_per_program must be4,8 or16")
    if tokens:
        _direct_g32_fp8_input[(triton.cdiv(tokens,rows_per_program),words)](
            x,out_q,out_sf,tokens,hidden,*x.stride(),*out_q.stride(),*out_sf.stride(),
            rows_per_program,num_warps=4,enable_fp_fusion=False)


# Short alias for callers that treat this as a direct producer into SymmBuffer.
cast_input_into = quantize_input_g32_into
