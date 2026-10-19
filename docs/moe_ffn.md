# INT8 MoE FFN

`grain_gemm.moe_ffn` runs a complete, pre-routed mixture-of-experts FFN. The
caller supplies floating-point tokens, prequantized expert weights, their scales
and top-k routing results. Routing preparation, input quantization, gate/up GEMM,
SwiGLU, intermediate quantization, down GEMM and weighted combination are included
in the call. Router logits and top-k selection remain outside it.

This is an inference function. The BF16/FP16 module that prepares and caches
quantized weights is a separate follow-up; this function does not re-quantize or
copy fixed expert weights on each forward.

## API

```python
from grain_gemm import moe_ffn

y = moe_ffn(
    hidden_states,
    w_gate_up,
    w_down,
    topk_weights,
    topk_ids,
    scale_gate_up=scale_gate_up,
    scale_down=scale_down,
    group_size=256,
    activation="swiglu",
    token_mask=None,
    backend="auto",
)
```

`T` is the number of tokens, `E` the number of experts, `H` the hidden size,
`I` the intermediate size, and `R` the number of expert slots per token.
The first implementation requires `G=256`, positive H/I divisible by 256,
positive E, and `1 <= R <= E`. T may be zero.

| Tensor | Shape | Dtype | Stride |
|---|---|---|---|
| `hidden_states` | `[T,H]` | BF16 or FP16 | `(H,1)` |
| `w_gate_up` | `[E,H,2I]` | INT8 | `(H*2I,1,H)` |
| `w_down` | `[E,I,H]` | INT8 | `(I*H,1,I)` |
| `topk_weights` | `[T,R]` | FP32 | `(R,1)` |
| `topk_ids` | `[T,R]` | INT32 or INT64 | `(R,1)` |
| `scale_gate_up` | `[E,H/G,2I]` | FP32 | `((H/G)*2I,2I,1)` |
| `scale_down` | `[E,I/G,H]` | FP32 | `((I/G)*H,H,1)` |
| `token_mask` | `[T]` or `None` | bool | `(1,)` |
| output | `[T,H]` | Same as input | `(H,1)` |

All tensors must be on the same SM80-or-newer CUDA device. Strides on dimensions
with size one may differ, since those dimensions are never stepped through;
this matches PyTorch's contiguous tensor behavior. Actual strides must remain
positive. Other layout, shape and dtype mismatches raise errors rather than
silently packing or transposing weights. Tensor and expanded-workspace indexing
is limited to signed 32-bit offsets.

The weight convention is exactly the existing GEMM convention, with an expert
dimension added. Start with contiguous Linear-format weights `[E,2I,H]` and
`[E,H,I]`, then pass `transpose(-2,-1)` views. This transpose only changes tensor
metadata. Their reduction dimensions remain contiguous in storage.

The first I output channels of `w_gate_up` are gate and the next I are up.
For example, the reconstructed gate/up weight is
`weight[e,n,k] = w_gate_up[e,k,n] * scale_gate_up[e,k//256,n]`.
There is no bias or shared-expert branch in this first API.

## Numerical and routing behavior

- Input activations are quantized independently per token and 256-element
  reduction group: `scale = absmax.clamp_min(1e-12)/127`, round to nearest even,
  clip to `[-127,127]`, and store INT8.
- Each GEMM uses INT32 dots within a group, FP32 `scale_a * scale_b`, and an
  FP32 fused multiply-add of the scaled group partial. Each projection's result
  is converted to the input dtype, as in the existing GEMM API.
- Gate/up results are converted to FP32 for `silu(gate) * up`, then quantized
  directly with the same group quantizer. There is no extra BF16/FP16 rounding
  of the SwiGLU product before quantization.
- Down-projection values are converted to FP32, multiplied by the supplied
  `topk_weights`, summed across slots in FP32 and converted to the input dtype.
  Scores are not softmaxed or renormalized. This ordering is important: applying
  scores before intermediate INT8 quantization would change the result.
- Every selected slot contributes, including repeated selections of an expert.
  Experts can have different token counts or receive no tokens. The function
  never drops routes to meet a fixed expert capacity.
- `token_mask=False` rows do not enter expert computation and return zero;
  `None` treats every row as valid. Masked expert ids are ignored. Counts and
  group boundaries are derived internally, so `num_tokens_per_expert` is not
  a public argument.
- Inputs, scales and scores are expected to be finite; valid expert ids are
  in `[0,E)`. Eager execution reads one small GPU statistics record to check
  expert ids and determine native capacity. Per-expert counts stay on the GPU.
  Graph replay requires callers to keep ids in range.

Synthetic quantization error relative to a BF16 FFN is separate from kernel
correctness. This API provides no autograd; use inference/no-grad mode when its
floating-point arguments would otherwise require gradients.

## Preparing explicit INT8 weights

The following is example preparation code, performed once outside forward.
It is not the proposed future BF16 module.

```python
import torch
from grain_gemm import moe_ffn


def prepare_weight(linear_weight):
    # Linear storage: [E,N,K], contiguous; logical GEMM weight: [E,K,N].
    e, n, k = linear_weight.shape
    blocks = linear_weight.float().reshape(e, n, k // 256, 256)
    scales = blocks.abs().amax(-1).clamp_min(1e-12) / 127
    q = (blocks / scales[..., None]).round().clamp(-127, 127).to(torch.int8)
    q = q.reshape(e, n, k).contiguous().transpose(-2, -1)
    scales = scales.transpose(-2, -1).contiguous()
    return q, scales


with torch.inference_mode():
    T, E, H, I, R = 32, 4, 512, 256, 2
    torch.manual_seed(2026)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(E, 2 * I, H, device="cuda", dtype=x.dtype) / H**0.5
    down = torch.randn(E, H, I, device="cuda", dtype=x.dtype) / I**0.5
    w1, s1 = prepare_weight(gate_up)
    w2, s2 = prepare_weight(down)
    probs = torch.randn(T, E, device="cuda").softmax(-1)
    scores, ids = probs.topk(R, dim=-1)
    scores = (scores / scores.sum(-1, keepdim=True)).contiguous()
    y = moe_ffn(x, w1, w2, scores, ids,
                scale_gate_up=s1, scale_down=s2)
    print(y.shape, y.dtype)  # [T,H], BF16
```

## Backends and CUDA Graphs

`backend="triton"` uses device-side routing and ragged grouped INT8 GEMMs. It
does not launch a Python expert loop. Its workspaces have fixed shapes based on
T/E/R; the routing metadata and valid tile counts are recomputed on the device.
This permits graph replay with changed tokens, ids, scores, masks and expert
counts. Warm up all signatures before capturing.

`backend="cutlass"` uses the native SM110 batched G256 kernel for both
projections. Actual per-expert counts are padded internally to a common
capacity. Weights and scales must be 16-byte aligned, BF16 activations are
required, and the matching optional library must be built:

```bash
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
```

The Thor native implementation adapts the G256 mainloop from
[`nvidia-cosmos/cosmos-int8-gemm`, commit `8977529`](https://github.com/nvidia-cosmos/cosmos-int8-gemm/blob/8977529aa23f70f02bfc36559599f00d59792996/docs/quantization/tools/cutlass_int8_sm110/blockwise_gemm.cuh#L79).
Cosmos configs 20/21 correspond to GrainGEMM configs 32/33: a 2SM
256x256x256 tile, TMA operands, INT32 TMEM partials, eight promotion warps,
three fragment buffers and vector scale loads. Config 32 retains the original
automatic epilogue tile; config 33 uses an explicit 128x16 tile. Both projections
use one batched launch over experts rather than one launch per expert.
The build enables `G128_OPT_PROMOTION`, `G128_OPT_PIPE2`, `G128_OPT_SBVEC` and
`NDEBUG`, as in the Cosmos production path. The activation-scale layout is
specialized to K-major so it directly consumes GrainGEMM's `[E,M,K/256]` scales;
weight scales remain `[E,K/256,N]`, with N contiguous.

Cosmos supplies the equal-capacity batched GEMM. This function additionally
implements ragged routing, padding without token dropping, dynamic activation
quantization, fused SwiGLU/intermediate quantization and FP32 weighted
combination. Native execution does not generate the ragged GEMM's tile metadata;
if auto falls back to Triton for skewed routing, it generates those tiles then.

`backend="cuda"` uses the existing CuTe GEMMs with a fixed expert loop. On
other supported native architectures, explicit CUTLASS also uses its existing
per-expert GEMMs. These explicit paths provide native comparisons without
changing the original single-GEMM implementations or their configuration IDs.
They require BF16 and the corresponding architecture-specific build; they do
not silently execute Triton when explicitly requested.

`backend="auto"` uses Triton unless a matching Thor CUTLASS build is available
for BF16 and the nominal dimensions satisfy `T*R >= E*128`. This first threshold
includes masked slots. It retains ragged Triton execution if equal-capacity
padding would exceed twice the number of valid, unmasked assignments. This is
a conservative heuristic, not an offline-tuned optimal dispatcher.

Captured auto execution always selects Triton. Explicit native capture reserves
the worst-case capacity `round_up(T*R,64)` per expert to accommodate any valid
routing on replay. This can use substantial memory and unnecessary compute;
prefer auto/Triton for graphs with dynamic routing.

## Validation and measurements

```bash
PYTHONPATH=src python -m pytest -q tests/test_moe_ffn.py
PYTHONPATH=src python benchmarks/bench_moe_ffn.py --output /tmp/moe_ffn.json
```

The tests use independent CPU INT32 group dots and per-token/top-k-slot
evaluation, rather than sharing the implementation's expert sorting and padding.
They cover ragged/duplicate routes, empty experts, masks, zero scales, INT8
extrema, both input dtypes, output ordering, streams and dynamic graph replay.

The benchmark measures the entire FFN, including routing, dynamic activation
quantization and combination. Weight quantization is excluded. Its floating
baseline is an eager PyTorch per-expert FFN reference, not a tuned fused BF16 MoE
kernel. It records the clock, temperature, correctness sample and timing rounds;
the earlier square-GEMM reports use a different protocol.

A [Thor end-to-end report](../benchmarks/results/thor/moe_ffn.md) records the
complete FFN at a reduced 1.2 GHz GPU clock (actual 1197 MHz), including effective
throughput and the precise eager BF16 baseline.
