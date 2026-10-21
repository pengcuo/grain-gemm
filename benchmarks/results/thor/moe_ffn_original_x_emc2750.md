# Thor MoE: read original INT8 tokens inside Gate/Up

The native Thor FFN now starts directly from the caller's original
`x_int8[T,H]` and FP32 `scale_x[T,H/256]`. Routing sorts token/rank indices only.
A dedicated producer warp loads selected original rows into shared memory with
vectorized `cp.async`. **No expert-expanded global input or input-scale tensor
is allocated or initialized, and no input pack kernel runs.** This applies to
all eligible Thor native shapes/top-k, including explicit native CUDA Graph
capture. Auto falls back to raw-token ragged Triton when the optional build is
absent, routing is too skewed or original x is not 16-byte aligned. Explicit
native calls reject missing builds or unaligned original x without copying.

Gate/Up, both BF16 rounding points before SwiGLU, FP32 SwiGLU and G256 hidden
quantization run in the same kernel. This also removes the full BF16 Gate/Up
global workspace for the large/uneven cases that previously used separate G1
and activation. Down GEMM and final weighted combination are still separate;
this is not a complete single-kernel INT8 Mega MoE.

## Same-run complete FFN results

GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**;
EMC was locked at **2750 MHz**. All six min/max/current readings were checked;
no clock settings were changed. Maximum sampled temperature was **62.843°C**.
Cooling starts at 62°C and execution aborts at 70°C.

E=32, H=1536, I=1024, Top-4, SwiGLU, no shared experts, one Thor GPU and no
communication. All methods share original BF16 values/weights, route IDs and
score values rounded through BF16. Both INT8 methods use the exact same
prequantized input/scale and weight/scale tensors. FP8 also starts with already
quantized activations/scales; entry quantization is outside all low-precision
timers. The uneven row has 16 experts with 4096 assignments each and 16 with
1000 each, for 20384 original tokens.

The before package was frozen immediately before this update. It is the prior
INT8-input API, including its input packer, counted native G2 and cached packed
G1 fusion at 4096/8192/16384; at 32768 and the uneven size it uses separate
native G1 and activation. Thus the before/after columns have the same input
format and timing boundary. All improvements are from this implementation
change, not moving entry quantization out of the timer.

| Input tokens | Packed INT8 before ms | Original-x INT8 ms (TOPS) | Latency reduction | FP8 Mega ms (TFLOPS) | Cosmos BF16 ms (TFLOPS) |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.293 | 3.191 (48.5) | 3.1% | 2.340 (66.1) | 5.528 (28.0) |
| 8192 | 5.434 | 5.128 (60.3) | 5.6% | 4.007 (77.2) | 9.413 (32.9) |
| 16384 | 9.511 | 9.097 (68.0) | 4.4% | 6.992 (88.5) | 16.784 (36.9) |
| 32768 | 18.202 | 17.471 (70.8) | 4.0% | 13.043 (94.8) | 31.798 (38.9) |
| 20384 (uneven) | 12.615 | 11.253 (68.4) | 10.8% | 8.488 (90.7) | 18.959 (40.6) |

The new path is faster on all five tested cases. Gains range from **3.1% to
10.8%**; it still does not match FP8 Mega performance. Savings in memory traffic
do not translate one-for-one into saved latency: G1 must perform indirect loads
and now includes the activation/quantization work at every measured size.
No additional optimization to Down or final combination is included here.

Each result is a complete eager FFN CUDA-event span, including validation,
GPU routing, the native CPU capacity read, runtime allocation, selected G1,
intermediate quantization, Down and combination. FP8 includes required route
ID/score copies and output allocation. Cosmos includes histogram, stable
sort/pad/gather, both `torch._grouped_mm` projections, activation/score-before-down
and BF16 scatter-add, compiled with default
`torch.compile(fullgraph=True,dynamic=True)`. Router logits/top-k, compilation,
entry/weight quantization, reusable FP8 buffer setup and correctness/cooling
are outside timing. No full-call CUDA Graph timing is used.

Two short warm calls per method precede four rounds of three samples. Order
rotates/reverses across four methods so every method occupies each position
equally. A common preallocated 256 MB L2 flush is outside every timer. Reported
latency is the median of round medians. Throughput counts useful operations
`6*T*4*1536*1024`, including Gate/Up and Down but excluding padding/non-GEMM
arithmetic; INT8 is TOPS and floating-point paths are TFLOPS.

## Removed global buffers

The input packer formerly wrote original activation values and scales to
`[E,C,H]` / `[E,C,H/256]`. These two allocations are gone. The second numeric
column below also records the BF16 `[E,C,2I]` allocation that disappears where
the predecessor had not already fused G1. Sizes are derived from actual case
capacities and tensor element sizes; they are not allocator-peak measurements
or measured DRAM traffic counters.

| Input tokens | C | Removed input/scale buffers | Additionally removed BF16 Gate/Up buffer |
|---:|---:|---:|---:|
| 4096 | 640 | 30.47 MiB | 0.00 MiB |
| 8192 | 1152 | 54.84 MiB | 0.00 MiB |
| 16384 | 2112 | 100.55 MiB | 0.00 MiB |
| 32768 | 4288 | 204.14 MiB | 536.00 MiB |
| 20384 (uneven) | 4096 | 195.00 MiB | 512.00 MiB |

At 32768 tokens this removes **204.14 MiB** of expanded input/scales and
**536 MiB** of BF16 Gate/Up storage. Avoided input-buffer writes are real, but
the same token is still needed by each selected expert: required original-x
reads are not claimed to disappear. Shared-memory caching and hardware caches
may reuse those reads; this report does not assert a DRAM-byte reduction factor.
QH/SH and Down output remain global buffers because those stages are separate.

## Implementation and validation

- `int8_g256_moe_gather.cu` extends the existing CUTLASS-derived paired two-SM
  pipeline. One warp loads original A, one loads weight B with TMA and one loads
  scales. Both CTA A producers signal the leader's barrier after their async
  loads complete; expected B transaction bytes and both A arrivals must complete
  before MMA. H1536 caches six A G256 blocks across the two I128 halves.
- Production selects the unrolled asynchronous vector-load variant. The CUDA
  compiler reports 96 registers/thread and 40-byte spill stores / 180-byte spill
  loads for CW16. A zero-spill bounded-loop variant and serial/parallel TMA
  gather4 alternatives were actually checked/timed; none beat the selected
  vector-load variant in the screened aggregate stage. They remain outside
  production. Removing spills alone did not improve that experiment's latency.
- Regression suite: **234 passed, 2 skipped** across six MoE test files.
  Coverage includes required prequantized API validation, offset views and
  source alignment fallback, masks/NaN scales, duplicate and invalid routes,
  signed/zero scales, empty experts, both output dtypes, nondefault streams,
  optional-build/ABI/hash checks, changing values/counts/scales on Graph replay
  and the native partial-padding scheduler.
- Actual public native tests replace both input packers with failure functions;
  execution succeeds. They assert that the G1 bridge receives the original x and
  scale pointer at 4096/8192/16384/**32768** tokens and compare all final output
  bits with the existing packed G1/activation/G2 reference.
- All five formal cases passed **full QH values, FP32 hidden scale bits, expert
  counts, padding and final BF16 output bits** versus the frozen predecessor.
  Independent quantized CPU checks and post-timing full-output repeat checks
  also passed. FP8 input values/scales match the upstream cast bitwise; Cosmos
  passes its independent BF16 CPU reference and repeated-output tolerance.
- Production-kernel Compute Sanitizer memcheck passed both CW8/CW16 at three
  shapes with R=1/3/4, H=256/512/1536 and I=256/512/1024, including empty experts,
  masked NaN scales, duplicates, full/tail clusters and graph replay after
  changing counts to all-zero and input scales to zero: **0 errors**.
- The extra non-even requested grid (3 physical CTAs, rounded to a two-SM pair
  grid) also passed the six CW8/CW16 sanitizer/replay cases with **0 errors**.
  Two infrastructure attempts stopped during Triton's `ldconfig` driver lookup
  before the gather kernel executed. Setting `TRITON_LIBCUDA_PATH=/lib` to the
  same installed CUDA driver bypassed that environment wrapper; the successful
  retry and both infrastructure logs are retained.
- Native/source/package/payload/script fingerprints remained unchanged through
  the complete formal measurement. Single-GEMM and SM120/GB10 native sources
  and dispatch tables were not changed.

DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`, vendored DeepGEMM
`1fc9c66975ee322d628b3288534ca0e9ce12a64d`, with the existing three host SM110
acceptance changes; device arithmetic/layout/scheduling remain unchanged.
FP8 uses E4M3 G32 UE8M0 and score-before-hidden-quantization/Down; INT8 retains
G256 FP32 scales and score-after-BF16-rounded Down. Cosmos Framework
`4ed6244506e00bb270ed17d4532897f2aaa7bc16` retains its grouped-MM BF16 numerical
contract. This compares performance, not equal model accuracy.

Build the new original-token kernel with:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /workspace/cutlass
python tools/build_moe_gather.py --cutlass-dir /workspace/cutlass
```

Raw formal samples, clocks/temperatures, correctness checks, test/sanitizer/build
logs, frozen packages, candidate sources/binaries/harnesses and SHA-256 manifest
are outside Git under `/workspace/artifacts/thor_moe_original_x_20261021/`.
