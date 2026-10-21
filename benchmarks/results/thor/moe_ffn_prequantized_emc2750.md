# Thor MoE: prequantized INT8-input API and FP8 comparison

The public `moe_ffn` now requires INT8 activations and their FP32 G256 scales.
It performs no input amax reduction, float conversion, quantization or scale
recomputation. This is an intentional API change: BF16/FP16 activation inputs
are rejected. Weight views/strides and weight scale layouts remain unchanged.
The new keyword-only arguments are required `scale_x` and optional
`output_dtype=torch.bfloat16`; FP16 output uses Triton, while native backends
require BF16 output. Both GEMM rounding points follow `output_dtype`.

```python
y = moe_ffn(
    x_int8, w_gate_up, w_down, topk_weights, topk_ids,
    scale_x=scale_x,                # FP32 contiguous [T,H/256]
    scale_gate_up=scale_gate_up,
    scale_down=scale_down,
    output_dtype=torch.bfloat16,
    backend="auto",
)
```

`x_int8[T,H]` has stride `(H,1)`, and `scale_x[T,H/256]` has stride
`(H/256,1)`, consistent with GEMM's activation/scale-A convention. All INT8
values, including -128, are consumed as supplied; scale values are copied
without recomputation. Native Thor scatters each token and scale to its expert
rows in a single launch, including zero-filled padding. Duplicate routes remain
separate assignments. Masked inputs and scales are not loaded. Other native
paths use the existing gather packer; ragged Triton consumes the original token
and scale tensors directly. SwiGLU/intermediate quantization remains inside the
FFN. No single-GEMM or SM120/GB10 native source was changed.

## Complete FFN comparison

GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**;
EMC was locked to **2750 MHz**. All six min/max/current readings were checked.
No clocks were changed. Maximum sampled temperature was **62.593°C**; cooling
starts at 62°C and execution aborts at 70°C.

E=32, H=1536, I=1024, Top-4, SwiGLU, no shared experts, single Thor GPU, no
communication. Original BF16 values/weights, routes and score values are shared.
Scores are FP32 rounded through BF16. The last row assigns 4096 rows to each
of 16 experts and 1000 rows to each of the other 16 experts.

**Current INT8 and FP8 both start with already quantized activations/scales.**
Their input quantizers run before the timer. FP8 input resides in its reusable
Mega buffer; its required ID/score copies and output allocation stay inside
timing. INT8's required expert packing, all validation/routing/capacity reads,
allocations, projections, middle activation/quantization and output combination
stay inside timing. FP8 Mega includes its dispatch, projections, middle
activation/quantization and combination. Cosmos starts with BF16 tokens.

The old INT8 control is the frozen package immediately before this API change;
it includes entry BF16 quantization and already has the latest padding skipping
and cached gate/up fusion. Its before/after improvement includes the change in
input/timing boundary and is not a same-input-format kernel speedup.

| Input tokens | Old BF16-input INT8 ms | New INT8-input FFN ms (TOPS) | Prequantized FP8 Mega ms (TFLOPS) | Cosmos BF16 ms (TFLOPS) | INT8 / FP8 time |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.350 | 3.287 (47.0) | 2.351 (65.8) | 5.562 (27.8) | 1.40× |
| 8192 | 5.521 | 5.359 (57.7) | 4.032 (76.7) | 9.505 (32.5) | 1.33× |
| 16384 | 9.747 | 9.411 (65.7) | 7.003 (88.3) | 16.813 (36.8) | 1.34× |
| 32768 | 18.685 | 18.169 (68.1) | 13.092 (94.5) | 32.197 (38.4) | 1.39× |
| 20384 (uneven) | 13.121 | 12.665 (60.8) | 8.510 (90.4) | 19.220 (40.0) | 1.49× |

All four methods are freshly measured in the same run: two short warm calls,
four rounds of three samples with balanced rotating/reversing order. A shared
preallocated 256 MB L2 flush runs before every timed call, outside the events.
Values are medians of round medians for complete eager CUDA-event spans,
including submission/synchronization gaps inside calls. No complete-FFN CUDA
Graph timing is used. Weight/input preparation, compilation, reusable Mega
workspace setup and router logits/top-k are excluded.

Throughput uses `6*T*4*1536*1024` useful projection operations, including gate/up
and down; padding and non-GEMM arithmetic are excluded. Entry quantization
exclusion does not remove the cost of expert packing or intermediate activations.
The earlier reports include entry quantization in FP8 as well as INT8; their
FP8 numbers are not reused for this new comparison.

## Isolated input stage

These separate measurements compare the old BF16 quantize-and-pack producer
against the new INT8/scale copy-and-pack producer at identical routes/capacities.
Every packed INT8 value and FP32 scale bit matches. They use short two-launch
CUDA graphs, three rounds of two samples and a 256 MB flush outside each sample.
This is a preallocated-input producer measurement, not the complete eager FFN
scope above; its timings should not be subtracted across measurement scopes.

| Tokens | Old quantize + pack ms | New pack only ms | Stage latency reduction |
|---:|---:|---:|---:|
| 4096 | 0.296 | 0.249 | 16.0% |
| 8192 | 0.553 | 0.445 | 19.7% |
| 16384 | 1.100 | 0.814 | 26.0% |
| 32768 | 2.055 | 1.631 | 20.6% |

At 32768 tokens, removing entry quantization reduces this stage by **0.424 ms**,
while packing still costs **1.631 ms**. The remaining middle activation and
output-combine passes are unchanged by this API update.

## Validation and numerical contracts

Regression coverage totals **215 passed, 2 skipped**: the main suite passed
209 tests with two skips, and all six new packing cases passed on their targeted
rerun after correcting the test's CPU/GPU comparison helper. The first main run
also passed all 201 then-present tests. No production correction was needed.
Coverage includes strict input/scale/output dtype/shape/stride validation,
-128 activation values, supplied signed/zero scales, masked NaN scales,
duplicate/invalid routes, zero tokens, both output dtypes, offset views, streams,
optional/fused dispatch and CUDA Graph replay with changing token values,
**input scale values**, counts, scores and masks. Entry quantizers are patched
to fail in public API tests, proving the new function does not call them.

Compute Sanitizer memcheck on all six new packing cases passed with **0 errors**.
The six cases exercise INT32/INT64 expert IDs, H=256/768/1536, unaligned original
input/scale pointers, signed/zero/NaN-masked scales and empty expert padding.

All five complete benchmark cases matched the former API's full QH bits,
intermediate FP32 scale bits, counts, padding and final BF16 output bits when
provided with the same quantized inputs. Both INT8 and FP8 passed independent
quantized CPU-reference checks before and after timing. FP8 input values/scales
matched the upstream cast bitwise. Cosmos passed its independent fused-pointwise
BF16 CPU reference and full repeated-output tolerance checks. Synthetic
common-FP32-reference differences are recorded; cross-format equality and equal
model accuracy are not claimed. Source/native/payload/script fingerprints were
stable throughout the formal comparison.

Cosmos Framework is the unchanged grouped-MM implementation at
`4ed6244506e00bb270ed17d4532897f2aaa7bc16`, using two `torch._grouped_mm`
calls under default `torch.compile(fullgraph=True,dynamic=True)`; its histogram,
sort/pad/gather, SwiGLU/score-before-down and BF16 scatter-add are included.
DeepInfer is `8b16149741a4935e1dec8cea27853f273b374c5b`, with DeepGEMM
`1fc9c66975ee322d628b3288534ca0e9ce12a64d` and the existing three host SM110
acceptance changes. Its device arithmetic/layout/scheduling are unchanged.
FP8 uses E4M3 G32 UE8M0 and scores before intermediate quantization/down;
INT8 retains G256 FP32 scales and scores after BF16-rounded down.

Raw measurements, frozen old/current packages, CPU/reference checks,
clock/temperature observations, test and memcheck logs, harnesses and SHA-256
manifest are outside Git under
`/workspace/artifacts/thor_moe_int8_input_20261021/`.

## Fresh repeat requested after the API update

A separate complete rerun uses the same weights, seeds, routing, prequantized
INT8/FP8 timing boundaries and four-round / three-sample protocol above. All
five cases again passed full INT8 before/after output and QH/scale-bit checks,
independent CPU-reference checks, and post-timing repeat checks. Source, package,
script and native fingerprints remained unchanged. GPU min/max/current stayed
at **1197 MHz**, with EMC min/max/current at **2750 MHz**; the nominal GPU
frequency remains **reduced to 1.2 GHz**. Maximum sampled temperature was
**63.093°C**, with the same 62°C cooling trigger and 70°C abort threshold.

| Input tokens | INT8-input FFN ms (TOPS) | Prequantized FP8 Mega ms (TFLOPS) | Cosmos BF16 ms (TFLOPS) | INT8 / FP8 time |
|---:|---:|---:|---:|---:|
| 4096 | 3.299 (46.9) | 2.347 (65.9) | 5.557 (27.8) | 1.41× |
| 8192 | 5.303 (58.3) | 4.003 (77.3) | 9.412 (32.9) | 1.32× |
| 16384 | 9.459 (65.4) | 7.019 (88.1) | 16.794 (36.8) | 1.35× |
| 32768 | 18.206 (67.9) | 13.048 (94.8) | 31.705 (39.0) | 1.40× |
| 20384 (uneven) | 12.609 (61.0) | 8.476 (90.8) | 18.833 (40.9) | 1.49× |

INT8 is **1.49–1.78× faster** than compiled Cosmos BF16 across these five cases;
FP8 Mega still takes less time. These are new complete eager FFN measurements,
not reused timings or standalone GEMM throughput. Both quantized paths exclude
entry quantization and include intermediate activation/quantization plus their
required dispatch/packing and output combination. The uneven distribution is
16 experts with 4096 rows each and 16 with 1000 each, as above.

The independent repeat's raw JSON, CSV and log are in the artifact bundle's
`repeat/` directory. Original measurements above are preserved separately.
