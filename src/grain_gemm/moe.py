# SPDX-License-Identifier: Apache-2.0
"""Inference-only INT8 MoE FFN with the same logical weights/scales as GEMM."""

import torch


def _stride(shape, column_major=False):
    if column_major:
        e, k, n = shape
        return (k * n, 1, k)
    result = []
    product = 1
    for dimension in reversed(shape):
        result.append(product)
        product *= dimension
    return tuple(reversed(result))


def _validate(hidden_states, w_gate_up, w_down, topk_weights, topk_ids,
              scale_gate_up, scale_down, token_mask, group_size, activation,
              backend):
    if type(group_size) is not int or group_size != 256:
        raise ValueError("moe_ffn currently requires group_size=256")
    if activation != "swiglu":
        raise ValueError("moe_ffn currently requires activation='swiglu'")
    if backend not in ("auto", "triton", "cuda", "cutlass"):
        raise ValueError("backend must be 'auto', 'triton', 'cuda', or 'cutlass'")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a two-dimensional tensor [T, H]")
    if not isinstance(w_gate_up, torch.Tensor) or w_gate_up.ndim != 3:
        raise ValueError("w_gate_up must be a three-dimensional tensor [E, H, 2I]")
    if not isinstance(w_down, torch.Tensor) or w_down.ndim != 3:
        raise ValueError("w_down must be a three-dimensional tensor [E, I, H]")
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a two-dimensional tensor [T, R]")
    tokens, hidden = hidden_states.shape
    experts, weight_hidden, twice_intermediate = w_gate_up.shape
    intermediate = twice_intermediate // 2
    routes = topk_ids.shape[1]
    if (experts <= 0 or hidden <= 0 or twice_intermediate <= 0
            or twice_intermediate % 2 or weight_hidden != hidden):
        raise ValueError("w_gate_up must have shape [E, H, 2I] with positive E, H, I")
    if hidden % group_size or intermediate % group_size:
        raise ValueError("H and I must be positive multiples of group_size=256")
    if not 1 <= routes <= experts:
        raise ValueError("top-k R must satisfy 1 <= R <= E")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("hidden_states must have dtype torch.bfloat16 or torch.float16")
    specs = [
        ("hidden_states", hidden_states, (tokens, hidden),
         (torch.bfloat16, torch.float16), False),
        ("w_gate_up", w_gate_up, (experts, hidden, 2 * intermediate), (torch.int8,), True),
        ("w_down", w_down, (experts, intermediate, hidden), (torch.int8,), True),
        ("topk_weights", topk_weights, (tokens, routes), (torch.float32,), False),
        ("topk_ids", topk_ids, (tokens, routes), (torch.int32, torch.int64), False),
        ("scale_gate_up", scale_gate_up, (experts, hidden // group_size, 2 * intermediate),
         (torch.float32,), False),
        ("scale_down", scale_down, (experts, intermediate // group_size, hidden),
         (torch.float32,), False),
    ]
    if token_mask is not None:
        specs.append(("token_mask", token_mask, (tokens,), (torch.bool,), False))
    for name, tensor, shape, dtypes, column_major in specs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape} and strided layout")
        if tensor.dtype not in dtypes:
            raise TypeError(f"{name} must have dtype {dtypes}")
        wanted_stride = _stride(shape, column_major)
        # A singleton dimension is never stepped through. PyTorch may retain
        # its old stride even after contiguous(), e.g. [E,N,1].transpose(1,2).
        # Match GEMM's contiguous-scale behavior without copying such views.
        if (any(size > 1 and actual != wanted for size, actual, wanted in
                zip(shape, tensor.stride(), wanted_stride))
                or (tensor.numel() and any(s <= 0 for s in tensor.stride()))):
            raise ValueError(f"{name} must have stride {wanted_stride}, got {tensor.stride()}")
        if tensor.device != hidden_states.device or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on the same CUDA device as hidden_states")
        if tensor.numel() > 2**31 - 1:
            raise ValueError(f"{name} exceeds the supported 32-bit indexing range")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("moe_ffn is inference-only; use torch.no_grad() for tensors requiring gradients")
    if tokens * routes > 2**31 - 1:
        raise ValueError("the expanded token count exceeds the supported indexing range")
    if tokens * routes * max(hidden, 2 * intermediate) > 2**31 - 1:
        raise ValueError("the expanded MoE workspace exceeds the supported indexing range")
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 8:
        raise RuntimeError("moe_ffn requires an SM80 or newer CUDA device")
    return tokens, hidden, intermediate, experts, routes


def _native_available(backend, device):
    if backend == "cuda":
        from .kernels import cuda
        return cuda.is_available(device)
    from .kernels import cutlass
    return cutlass.is_available(device)


def _native_gemm(a, weight, sa, sb, out, backend):
    if backend == "cutlass" and torch.cuda.get_device_capability(a.device) == (11, 0):
        from .kernels import cutlass
        return cutlass.launch_batched(a, weight, sa, sb, out, config_id=32)
    from .gemm import get_kernel_config
    from .kernels import cuda, cutlass
    implementation = cuda if backend == "cuda" else cutlass
    # Other native targets retain their existing GEMM implementations. The
    # expert loop has a fixed length, not a device-to-host routing dependency.
    for expert in range(weight.shape[0]):
        b = weight[expert].t()
        config = get_kernel_config(
            a[expert], b, group_size=256, output_dtype=torch.bfloat16,
            scale_a=sa[expert], scale_b=sb[expert], backend=backend,
        )
        implementation.launch(a[expert], b, sa[expert], sb[expert], out[expert],
                              config["config_id"])
    return out


def moe_ffn(hidden_states, w_gate_up, w_down, topk_weights, topk_ids, *,
            scale_gate_up, scale_down, group_size=256, activation="swiglu",
            token_mask=None, backend="auto"):
    """Run a complete pre-routed INT8 G256 MoE FFN on the current CUDA stream.

    Weights follow GEMM's right-hand operand convention, not Linear's storage
    shape: w_gate_up[E,H,2I] and w_down[E,I,H] must be transposed views of
    contiguous Linear-format INT8 weights, with strides (H*2I,1,H)/(I*H,1,I).
    FP32 scales are contiguous [E,H/256,2I] and [E,I/256,H], like scale_b.
    The gate channels precede the up channels. Inputs [T,H] are BF16/FP16;
    route scores [T,R] are FP32, expert ids [T,R] are INT32/INT64. All other
    inputs require canonical contiguous strides and the same CUDA device.

    Input and FP32 SwiGLU activations are dynamically quantized per row/K group
    using amax.clamp_min(1e-12)/127, nearest-even rounding and clipping to
    [-127,127]. Each GEMM accumulates group dots in INT32 and scaled sums in
    FP32, then rounds to the input dtype. Scores multiply the second GEMM's
    outputs, followed by FP32 reduction and one cast to the input dtype.
    Scores are used as supplied, without renormalization. A token may select
    the same expert more than once; every slot contributes independently.

    Arbitrary expert counts, empty experts and mask=False rows are supported
    without token dropping. Masked rows return zero; None means all rows are
    valid. Routing and weights must be finite and valid ids lie in [0,E).
    Eager calls validate id values; graph replay requires callers to preserve
    this value contract. Warm up before graph capture to compile kernels.

    Auto uses the ragged Triton path, or the SM110 CUTLASS batched path for
    dense-enough BF16 routing. Captured auto calls use Triton so changing
    routing needs no host synchronization or fixed expert capacity. Explicit
    native capture reserves the worst-case per-expert capacity T*R (rounded
    to 64), which can be expensive for many experts. Native backends require
    BF16 and a matching optional build. No autograd is provided.
    """
    tokens, hidden, intermediate, experts, top_k = _validate(
        hidden_states, w_gate_up, w_down, topk_weights, topk_ids,
        scale_gate_up, scale_down, token_mask, group_size, activation, backend,
    )
    from .kernels import moe as kernels

    with torch.cuda.device(hidden_states.device):
        capturing = torch.cuda.is_current_stream_capturing()
        if tokens == 0:
            return torch.zeros_like(hidden_states)

        selected = backend
        if backend == "auto":
            selected = "triton"
            if (not capturing and hidden_states.dtype == torch.bfloat16
                    and torch.cuda.get_device_capability(hidden_states.device) == (11, 0)
                    and tokens * top_k >= experts * 128
                    and all(t.data_ptr() % 16 == 0 for t in (
                        w_gate_up, w_down, scale_gate_up, scale_down))
                    and _native_available("cutlass", hidden_states.device)):
                selected = "cutlass"

        routes = kernels.prepare_routes(
            topk_ids, experts, token_mask, include_tiles=selected == "triton",
        )
        if not capturing:
            # The routing kernels validate ids and reduce the native capacity
            # statistics together. One small read replaces a separate Torch
            # predicate/reduction and a second expert-count synchronization.
            invalid, maximum, total = routes.statistics.cpu().tolist()
            if invalid:
                raise ValueError("unmasked topk_ids must be in [0, E)")

        if backend in ("cuda", "cutlass"):
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError(f"backend='{backend}' requires BF16 hidden_states")
            if not _native_available(backend, hidden_states.device):
                raise RuntimeError(f"MoE backend='{backend}' needs a matching native build; "
                                   f"run tools/build_{backend}.py --cutlass-dir PATH --arch sm_XXX")
            if any(t.data_ptr() % 16 for t in (w_gate_up, w_down, scale_gate_up, scale_down)):
                raise ValueError(f"backend='{backend}' requires 16-byte-aligned weights and scales")

        capacity = None
        if selected in ("cuda", "cutlass"):
            if capturing:
                capacity = ((tokens * top_k + 63) // 64) * 64
            else:
                if maximum == 0:
                    return torch.zeros_like(hidden_states)
                capacity = ((maximum + 63) // 64) * 64
                # A heavily skewed route would otherwise compute E copies of
                # the busiest expert's capacity. Auto retains ragged execution.
                if backend == "auto" and experts * capacity > 2 * total:
                    selected = "triton"
                    capacity = None
                    routes = kernels.prepare_ragged_tiles(routes)

        qx, sx = kernels.quantize_rows(hidden_states, group_size)
        w1_linear = w_gate_up.transpose(-2, -1)
        w2_linear = w_down.transpose(-2, -1)

        if selected == "triton":
            gate_up = kernels.ragged_gemm(
                qx, w1_linear, sx, scale_gate_up, routes,
                output_dtype=hidden_states.dtype, source_is_token=True,
            )
            qh, sh = kernels.swiglu_quantize(gate_up, routes=routes)
            expert_output = kernels.ragged_gemm(
                qh, w2_linear, sh, scale_down, routes,
                output_dtype=hidden_states.dtype, source_is_token=False,
            )
        else:
            if max(experts * capacity * hidden,
                   experts * capacity * 2 * intermediate) > 2**31 - 1:
                raise ValueError("native MoE workspace exceeds the supported indexing range")
            packed_x, packed_sx = kernels.pack_inputs(qx, sx, routes, capacity)
            gate_up = torch.empty((experts, capacity, 2 * intermediate),
                                  device=hidden_states.device, dtype=hidden_states.dtype)
            _native_gemm(packed_x, w1_linear, packed_sx, scale_gate_up, gate_up, selected)
            qh, sh = kernels.swiglu_quantize(gate_up)
            expert_output = torch.empty((experts, capacity, hidden),
                                        device=hidden_states.device, dtype=hidden_states.dtype)
            _native_gemm(qh, w2_linear, sh, scale_down, expert_output, selected)
        return kernels.combine(expert_output, topk_weights, routes,
                               output_dtype=hidden_states.dtype, packed_capacity=capacity)
