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

For up to 64 experts, routing uses stable device-side counting sort: chunk
histograms, a prefix sum, then stable scattering. SM110 uses 2048-slot chunks
for up to 262144 expanded slots; other architectures retain 512-slot chunks and
the 65536-slot threshold. Larger cases retain the general stable-sort path.
Quantization and packing process several rows per program; output combination
uses wider column tiles to amortize routing-index loads. Every path preserves
duplicate slots and masks.

`backend="cutlass"` uses native SM110 G256 kernels for the expert
projections. Actual per-expert counts are padded internally to a common
capacity. Weights and scales must be 16-byte aligned, BF16 activations are
required, and the matching optional library must be built:

```bash
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
```

The Thor build also installs a separate `libgrain_cutlass_moe.so`. To rebuild
only the MoE kernels without rebuilding single GEMM:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /path/to/cutlass
```

The Thor native implementation adapts the G256 mainloop from
[`nvidia-cosmos/cosmos-int8-gemm`, commit `8977529`](https://github.com/nvidia-cosmos/cosmos-int8-gemm/blob/8977529aa23f70f02bfc36559599f00d59792996/docs/quantization/tools/cutlass_int8_sm110/blockwise_gemm.cuh#L79).
Cosmos configs 20/21 correspond to the original GrainGEMM configs 32/33: a 2SM
256x256x256 tile, TMA operands, INT32 TMEM partials, eight promotion warps,
three fragment buffers and vector scale loads. Config 32 retains the original
automatic epilogue tile; config 33 uses an explicit 128x16 tile. These remain
the compatibility path when the separate MoE library is unavailable.
The build enables `G128_OPT_PROMOTION`, `G128_OPT_PIPE2`, `G128_OPT_SBVEC` and
`NDEBUG`, as in the Cosmos production path. The activation-scale layout is
specialized to K-major so it directly consumes GrainGEMM's `[E,M,K/256]` scales;
weight scales remain `[E,K/256,N]`, with N contiguous.

The separate MoE library reuses that collective with 128-row and 256-row tiles,
one-SM/two-SM variants and dynamic/static persistent schedulers. Counts-aware
variants read `[E]` INT32 counts on the current CUDA stream and skip every
work tile whose entire cluster lies in padding, including initial work and
empty experts. A cluster with no live work returns before pipeline setup;
otherwise every producer, consumer and peer CTA starts at the same first live
task. Skipped tiles perform no GEMM operand loads, matrix arithmetic or output
stores. Partially live clusters still compute their full 128-row or 256-row
tile. Valid rows retain the original G256 math; padding output need not be
written. SwiGLU
masks these rows before loading, and combination reads only live routes.
Entirely padded SwiGLU blocks write zero INT8 values and the original finite
zero-group scale directly, avoiding nonlinear math while fully initializing
the next GEMM's inputs.
Each projection uses a single batched launch over experts. All native workspace
belongs to the invocation; no tensor pointers or routing-dependent arguments
are cached between calls or graph replays.

For Thor native top-k up to eight, a fused producer quantizes each original
token/group once and scatters it to its selected packed expert rows. A separate
padding kernel initializes only unused rows. This removes intermediate token
INT8/scale tensors while retaining the original packed layout, exact rounding,
duplicate-route behavior and fully initialized TMA inputs.
For the calibrated E32/H1536/Top-4 cases with at least 16384 tokens, a separate
input producer shares route loads across adjacent G256 groups and initializes
padding in the same launch.

An additional optional SM110 library fuses gate/up projection, FP32 SwiGLU and
G256 activation quantization. Build it independently of either GEMM library:

```bash
python tools/build_moe_fused.py --cutlass-dir /path/to/cutlass
```

This installs `libgrain_moe_fused.so` and `moe_fused_build.json` under the private
native-library directory. The loader checks the SM110 target, ABI, source hashes
and binary hash. A missing or stale build retains the original gate/up GEMM
plus activation-quantization path. No new public arguments or weight preparation
are required.

Eligible eager CUTLASS calls, including `backend="auto"` after it selects
CUTLASS, use the fusion for BF16 inputs with T=4096, 8192 or 16384 and
E=32/H=1536/I=1024/Top-4. Existing padding/skew checks still apply. T=32768
and other token counts retain the native separate gate/up GEMM and activation
quantization. SM120/GB10, FP16 auto, captured auto and captured explicit native
calls retain their existing implementations.

The fused kernel uses two-SM clusters with a 128-row cluster tile, 64 rows per
physical CTA, and paired 128-channel gate/up projections. Each G256 output
group is computed in two I128 halves. Gate and up share the loaded input A;
for H=1536, all six input G256 blocks remain in shared memory for reuse by the
second I128 half. Two operand stages and four paired TMEM accumulator buffers
overlap input movement, INT8 matrix operations and FP32 scaled accumulation.
The private launcher's `persistent_blocks` counts physical CTAs and rounds up
to an even count. Zero starts from the device SM count; the number of clusters
is capped by the available cluster tasks.

Both projections retain their BF16 rounding before FP32 SwiGLU. The complete
256-channel SwiGLU group stays FP32 in CTA shared memory until its scale
reduction and INT8 quantization, then the kernel writes INT8 `[E,C,I]` and FP32
`[E,C,I/256]` directly. It removes the full BF16 `[E,C,2I]` global intermediate
allocation and its write/read. Packed inputs and quantized intermediate
outputs still use global memory; input packing, down GEMM and output
combination remain separate kernels. This is a fused G1 stage, not a complete
single-kernel Mega MoE. Public weight/scale shapes, strides, G256 grouping and
score placement are unchanged. Kernel arguments and output buffers belong to
each invocation on its current stream; weight tensors and their pointers are
not cached.

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

Within the separate Thor MoE library, the default is config 43: a counts-aware
static persistent 2SM 128x256x256 tile. Full-FFN calibration selects config 44, a
counts-aware static 1SM 128x128x256 tile, for `(E,C,N,K)` equal to
`(32,320,2048,1536)` or `(32,320,1536,1024)`. Config 47 uses a counts-aware
static 2SM 256x256x256 tile and is selected for measured E32 capacities:

| Projection `(N,K)` | Capacities selecting config 47 |
|---|---|
| Gate/up `(2048,1536)` | 768, 960, 1024, 1088, 1152, 1216, 2048, 2112, 2176, 2240, 4096, 4288, 4352 |
| Down `(1536,1024)` | 768, 960, 1024, 1216, 2048, 2240, 4096, 4352 |

Measured exceptions take priority over that table: E32 down `(N,K)=(1536,1024)`
at capacities 640, 1152 and 4288 selects config 44; the first two use swizzle 4.
E32 gate/up `(N,K)=(2048,1536)` at capacity 4288 also selects config 47.
Uncalibrated shapes retain config 43. Config 46 provides a counts-aware static
2SM 256x128x256 alternative. Original dynamic configurations 34, 36 and 45
remain available for isolated comparisons. The projections choose independently;
these choices
do not modify any single-GEMM dispatch table. Captured explicit native calls
still use their reserved worst-case capacity when choosing a configuration.

Captured auto execution always selects Triton. Explicit native capture reserves
the worst-case capacity `round_up(T*R,64)` per expert to accommodate any valid
routing on replay. Counts-aware tiles reread the GPU counts on every replay
and skip wholly unused clusters, but the reserved memory and partial-cluster
padding remain. Input/intermediate padding initialization still writes its
required zero values. Prefer auto/Triton for graphs with dynamic routing.
See the [per-expert padding-skip measurements](../benchmarks/results/thor/moe_ffn_padding_skip_emc2750.md).

## Validation and measurements

```bash
PYTHONPATH=src python -m pytest -q tests/test_moe_ffn.py
PYTHONPATH=src python -m pytest -q tests/test_moe_fused.py tests/test_moe_fused_sm110.py
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

A [Cosmos Framework comparison](../benchmarks/results/thor/moe_ffn_cosmos.md)
measures the complete pre-routed FFN against its original grouped-MM BF16 path,
both eager and compiled, at a reduced 1.2 GHz GPU clock (actual 1197 MHz).
The [earlier report](../benchmarks/results/thor/moe_ffn.md) retains the separate
per-expert-loop BF16 baseline; its speedups do not describe Cosmos grouped-MM.

An additional [BF16 comparison for the upstream vLLM expert core used by
Omni](../benchmarks/results/thor/moe_ffn_vllm.md) measures the original Triton
functional FFN and freshly remeasured Cosmos compiled FFN on the same inputs
at 1197 MHz. Its scope includes pre-routed dispatch and both projections,
without the full Omni serving engine.

The [optimized INT8 comparison](../benchmarks/results/thor/moe_ffn_optimized.md)
remeasures before/after INT8, upstream vLLM BF16 and Cosmos compiled BF16 in
one process on identical inputs at 1197 MHz. It records actual per-projection
native configuration IDs, independent correctness checks, raw samples and
source/build hashes; the earlier measurements above remain unchanged.

The [cached two-SM benchmark](../benchmarks/results/thor/moe_ffn_cached_2sm_emc2750.md)
records the current implementation against the immediately preceding package
and DeepInfer FP8 at 1197 MHz GPU / 2750 MHz EMC. It includes public eager
timings, stage breakdowns, bit comparisons and rejected full-fusion prototypes.

The [earlier SM110 fused INT8 and DeepInfer FP8 comparison](../benchmarks/results/thor/moe_ffn_mega_int8_emc2750.md)
records the original one-SM fused implementation, selected at T=4096/8192,
using actual before/current public auto calls at a reduced nominal 1.2 GHz GPU
clock (1197 MHz) with EMC locked at 2750 MHz. Its source hashes, dispatch scope
and measurements describe that historical run, preceding the cached two-SM
implementation above. It includes exact intermediate/output bit checks and
the complete eager FFN. DeepInfer FP8 remains a performance reference with a
different quantization and score-placement contract.
