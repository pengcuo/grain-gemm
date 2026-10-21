# Interleaved gate/up weights

`moe_ffn(..., gate_up_layout="interleaved")` accepts the first projection's
output channels in this order:

```text
g0, u0, g1, u1, ..., g(I-1), u(I-1)
```

The default `gate_up_layout="split"` retains the existing
`g0, ..., g(I-1), u0, ..., u(I-1)` ordering. The function consumes both layouts
directly; it does not copy or repack weights or their scales during a call.

The logical mathematical operands and required strides are unchanged:

| Argument | Shape | Stride |
| --- | --- | --- |
| `w_gate_up` (INT8) | `[E,H,2I]` | `(H*2I,1,H)` |
| `scale_gate_up` (FP32) | `[E,H/256,2I]` | `((H/256)*2I,2I,1)` |
| `w_down` (INT8) | `[E,I,H]` | `(I*H,1,I)` |
| `scale_down` (FP32) | `[E,I/256,H]` | `((I/256)*H,H,1)` |

Weight output column `2*j` is gate channel `j`; column `2*j+1` is up channel
`j`. **Apply exactly the same output-column permutation to
`scale_gate_up`.** Its reduction-group dimension remains `H/256`, and the
post-SwiGLU quantization groups still contain 256 consecutive hidden channels.
Down weights and scales retain their original channel order.

For example, prepare an existing split checkpoint once, outside timed forward
calls, preserving the transposed Linear-weight view:

```python
# w_gate_up: [E,H,2I], stride (H*2I,1,H)
# scale_gate_up: contiguous [E,H/256,2I]
i = w_gate_up.shape[-1] // 2
channels = torch.arange(i, device=w_gate_up.device)
order = torch.stack((channels, channels + i), dim=-1).reshape(-1)
linear_weight = w_gate_up.transpose(-2, -1)  # contiguous [E,2I,H]
w_gate_up_interleaved = linear_weight.index_select(1, order).transpose(-2, -1)
scale_gate_up_interleaved = scale_gate_up.index_select(-1, order).contiguous()

y = moe_ffn(
    hidden_states, w_gate_up_interleaved, w_down, topk_weights, topk_ids,
    scale_x=scale_x,
    scale_gate_up=scale_gate_up_interleaved,
    scale_down=scale_down,
    gate_up_layout="interleaved",
)
```

On Thor SM110, the optional `tools/build_moe_interleaved.py --cutlass-dir PATH`
build implements original-token gather, gate/up GEMM, SwiGLU and intermediate
INT8 quantization in one first-projection launch. It writes the quantized
intermediate and its scales directly, without a full global gate/up tensor.
The split build remains independent. Auto dispatch requires the selected
layout's build; otherwise it uses the layout-aware Triton path. FP16 output,
captured auto calls and heavily skewed routing also retain the Triton path.
Explicit native calls require the corresponding native build and BF16 output.

The two layouts compute the same `SiLU(gate[j]) * up[j]`, with the existing
BF16/FP16 first-GEMM rounding and FP32 activation/quantization semantics.
Interleaving alone does not remove global traffic from a separate activation
kernel; the native first-projection fusion supplies that reduction. The existing
split native gather path already fuses these stages, so the comparison between
the native layouts measures their kernel organization, not an additional full
gate/up write/read removed only by interleaving.
