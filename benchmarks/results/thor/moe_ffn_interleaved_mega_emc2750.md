# Thor: interleaved INT8 Mega FFN attempt

The new explicit SM110 implementation fuses original-token gather, interleaved
Gate/Up, exact SwiGLU, G256 hidden quantization and Down in one persistent
kernel. The measured version keeps a separate weighted output combine. Complete
FFN latency is **2.1–9.0% lower** than the immediately preceding interleaved
INT8 implementation in this run, and remains **15.7–29.8% higher** than the
DeepInfer FP8 reference. This is an experimental path with a narrow shape
contract; the production dispatcher and GB10 implementation are unchanged.

## Complete FFN measurements

GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**.
EMC was **2750 MHz**. Every accepted call checked all six min/max/current clock
readings. In the latest INT8/FP8 run, the maximum sampled temperature was **62.468 C**;
execution cools from 62 C to 54 C and aborts at 70 C. An observed EMC reset
caused setup attempts to be rejected before timing; the previously requested
2750 MHz setting was reapplied through the same-kernel host. Those rejected
attempts are excluded from this table and retained in the external archive.

E=32, H=1536, I=1024, Top-4, SwiGLU, one Thor GPU, no shared experts and no
communication. Both INT8 implementations receive exactly the same supplied
INT8 activations, FP32 G256 scales, weights, route IDs and scores. Input/weight
quantization and the one-time interleaving permutation are outside timing for
all low-precision methods. FP8 starts from its upstream-compatible quantized
input too. Original BF16 values/weights/routes are shared across formats.

| Tokens | Previous INT8 ms (TOPS) | Mega INT8 ms (TOPS) | Latency reduction | DeepInfer FP8 ms (TFLOPS) | Cosmos BF16 ms (TFLOPS), historical | Mega / FP8 time |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2.985 (51.80) | **2.924 (52.88)** | 2.1% | 2.252 (68.66) | 5.528 (27.97) | 1.298x |
| 8192 | 4.972 (62.19) | **4.524 (68.36)** | 9.0% | 3.912 (79.06) | 9.413 (32.85) | 1.157x |
| 16384 | 8.622 (71.73) | **8.410 (73.54)** | 2.5% | 6.955 (88.93) | 16.784 (36.85) | 1.209x |
| 32768 | 16.579 (74.61) | **16.151 (76.59)** | 2.6% | 13.417 (92.19) | 31.798 (38.90) | 1.204x |

**BF16 was not rerun with this Mega experiment.** Its column is reused from the
[earlier complete FFN measurement](moe_ffn_original_x_emc2750.md) at the same
E/H/I/Top-k, GPU and EMC settings. That run used four rounds of three samples,
the median of round medians and the same 256 MB flush policy. Cosmos BF16
includes routing/counts, stable sort/pad/gather, both `torch._grouped_mm`
projections, SwiGLU with scores before Down, and BF16 scatter-add, compiled with
`torch.compile(fullgraph=True, dynamic=True)`. It starts from BF16 activations;
low-precision entry quantization is excluded as described above. These are
complete FFN results, not standalone BF16 GEMM throughput. The BF16 column
provides historical context rather than a four-way same-run comparison.

For a fresh four-way run including DeepInfer **Mega BF16**, see the
[Mega BF16 comparison](moe_ffn_mega_bf16_emc2750.md).

Throughput counts **all three projections**, `4*T*R*H*I + 2*T*R*I*H`, or
`6*T*R*H*I`, using useful token assignments. Padding and non-GEMM arithmetic
are excluded from the numerator. Integer throughput is TOPS; floating-point
throughput is TFLOPS.

The latest INT8/FP8 columns use full eager CUDA-event spans, three rounds of two samples per method,
with balanced rotating/reversing order and a common 256 MB flush outside each
timer. Reported latency is the median of round medians. Full-call CUDA Graphs
are not used. INT8 retains public tensor validation, routing, the host capacity
statistics read, per-call allocations, the selected core and output combine.
FP8 includes route ID/score copies, output allocation and its complete device
Mega operation, while its reusable SymmBuffer setup remains outside timing.
This compares the actual APIs and allocation policies, not pure GEMM kernels or
an isolated format conversion. Small gains, especially at 4096 tokens, are
sensitive to host submission/synchronization variation.

## What changed and which IO remains

The weight and scale input convention is unchanged: logical mathematical
weights `[E,H,2I]` and `[E,I,H]` are transposed Linear-storage views; output
columns of Gate/Up **and their scales** use `[g0,u0,g1,u1,...]`. No per-call
weight packing or input/weight scale recomputation occurs. Hidden activation
scales are still computed inside the core on every forward call.

A persistent two-SM cluster owns an expert M128 token tile. It keeps quantized
hidden values and scales on chip between the two projections. The core reuses
dead cached Gate/Up input storage for a 32 KiB swizzled BF16 Down output tile,
then copies contiguous rows with aligned 16-byte stores. This preserves the
GEMM rounding points and improves the otherwise strided epilogue.

The predecessor already fused Gate/Up, activation and hidden quantization;
there was **no full BF16 gate/up global buffer** left to remove. This attempt
additionally removes the global QH/SH allocations and their producer/consumer
traffic. Useful QH+SH write/read volume is 32.5 MiB at 4096 tokens and 260 MiB
at 32768, excluding padding. These are logical tensor bytes, not measured DRAM
counters. BF16 expert outputs and the final weighted combine still use global
memory. Scratch staging adds shared-memory traffic rather than eliminating all
memory operations.

The selected build also uses an exact 256 KiB SiLU table. Gates are BF16-rounded,
so all 65536 possible patterns can be evaluated once with the original FP32
`g/(1+expf(-g))`. Lookup preserves the original result; table generation is
independent of runtime tokens/weights and is initialized before warmup. Table
reads still consume cache bandwidth. G256 integer quantization uses a guarded
reciprocal and falls back to the original rounded division near integer-rounding
ties; scales remain unchanged. See the
[rounding proof](../../../experiments/thor_interleaved_mega/quantization_proof.md).

## Separate stage profile and remaining gap

The following is a **separate Nsight run**, using three calls per method.
Durations are additive means, and throughput in parentheses is recomputed from
the same mean. They are not subtracted from the primary timing table above.
At 32768 tokens:

| Stage | Previous INT8 ms | Mega INT8 ms | DeepInfer FP8 ms |
|---|---:|---:|---:|
| Routing/statistics copy or FP8 route copy | 0.125 | 0.123 | 0.032 |
| Gate/Up + activation/quantization | 9.529 (86.5 TOPS) | Included in core | Included in Mega |
| Down | 3.566 (115.6 TOPS) | Included in core | Included in Mega |
| Fused core / complete FP8 Mega | — | 12.705 (97.4 TOPS) | 12.924 (95.7 TFLOPS) |
| Separate output combine | 2.997 | 2.997 | Included in Mega |
| Time outside GPU activities | 0.441 | 0.413 | 0.113 |
| Full profiled call | 16.658 | 16.238 | 13.069 |

The INT8 core and the FP8 Mega column cover different work: FP8 already includes
its output combination. The roughly 3 ms separate INT8 combine is a real cost,
but its numerical similarity to the total FP8 gap does **not** establish that
all of that gap can be removed by moving it inside a kernel. FP8 also writes and
reads BF16 slot outputs. The separate profile did not reproduce the small 4096
gain: INT8 baseline/new means were 3.264/3.417 ms, versus 2.985/2.924 ms in the
primary non-profiled experiment. Full profile samples for both sizes are archived.

Source inspection identifies implementation differences that interleaving alone
cannot resolve:

- DeepInfer uses hardware block-scaled FP8 MMA and retains its K reduction in
  FP32 tensor memory. INT8 G256 reads each INT32 partial, converts it, applies
  FP32 A/B scales and accumulates after every K256 group. This adds software
  promotion and synchronization without adding useful GEMM FLOPs.
- Its swapped-operand token tiles use M192/M240, six operand stages and adaptive
  tails. This INT8 core uses M128/N256 and two stages, and completes the whole
  expert tile's Gate/Up before Down. A tested M256 Down repartition and four-stage
  variant did not improve the complete call.
- DeepInfer schedules independent tile dependencies. Its final output publication
  uses one grid-barrier atomic per persistent CTA and a rank readiness flag, then
  a dedicated in-kernel combine phase. It does not require per-token output-ready
  atomics, and it does not make slot IO disappear. All of this remains single-card
  work in the reference; communication is not the explanation for this gap.

These are concrete source differences, not a measured stall attribution or a
proof of an INT8 performance ceiling.

## Output-combine fusion trials

Two further kernels preserved full QH/SH, useful BF16 expert output, final output
bits and exact four-contribution counters in the tested 4096/32768 cases. They
were not selected. Each row below is a separate paired pilot, three rounds of
one sample; the paired baseline is shown to avoid mixing runs.

| Publication strategy | Tokens | Paired previous INT8 ms | Trial ms |
|---|---:|---:|---:|
| Per-token/N256 last contributor | 4096 | 2.894 | 3.339 |
| Per-token/N256 last contributor | 32768 | 16.476 | 18.763 |
| Once per token after all Down outputs | 4096 | 3.054 | 2.959 |
| Once per token after all Down outputs | 32768 | 16.482 | 16.674 |

The second trial reduced publication from six counters per token to one and
batched the device fences. It recovered much of the first trial's regression,
but did not beat the selected core on the larger case. Both used deterministic
slot-order FP32 arithmetic, GPU release/acquire publication and coherent slot
loads; neither used floating-point atomic output accumulation or inter-CTA
spinning. These publication strategies did not produce a consistent full-FFN win;
the trials do not isolate their individual stall or scheduling costs.
They remain in the external experiment archive rather than the shipped default.

## Correctness, use and provenance

- The selected library passed 24 targeted edge combinations: CW8/CW16, default
  and alternate streams, partial tiles, INT8 -128, signed/zero scales, duplicate
  routes, masked invalid IDs/NaN input scales, empty experts and empty inputs.
- Compute Sanitizer memcheck passed those 24 cases with **0 errors**, using
  `--report-api-errors explicit` as in the preceding Thor checks. The initial
  default API-reporting run emitted the environment's TileIR-compiler-load
  diagnostic during CUDA initialization; its log is retained separately.
- All 65536 actual SiLU table patterns matched, including the initialization-event
  dependency across two streams. A separately compiled helper also checked all
  gates with 24 representative BF16 Up values and one million random pairs.
  The guarded quantizer passed 8,326,770 adversarial GPU cases with zero mismatches.
- Every formal case matched full QH bits, FP32 scale bits, useful BF16 expert-Y
  bits, padding, route counts and final BF16 output before/after timing. Separate
  INT8/FP8 CPU references were checked. Cross-format bit identity or equal model
  accuracy is not claimed.
- Independent CPU audits verified 72 timed calls, 144 sample clock endpoints,
  407 temperature/clock observations and 1320 source/native/payload hashes. All
  48 production files matched the pre-task snapshot. The portable benchmark's
  post-timing output smoke check also passed.

Use the [explicit experimental API and build instructions](../../../experiments/thor_interleaved_mega/README.md).
The measured build enables `--silu-lut`, CW16, H1536/I1024, G256, BF16 output
and eager SM110 execution. Final combine remains separate. The ordinary
`grain_gemm.moe_ffn` dispatcher is unchanged.

DeepInfer is `8b16149741a4935e1dec8cea27853f273b374c5b`, vendored DeepGEMM
`1fc9c66975ee322d628b3288534ca0e9ce12a64d`, with the existing three host SM110
acceptance changes; device arithmetic/scheduling is unchanged. FP8 uses E4M3
G32 UE8M0 and its upstream fast math/score-before-hidden-quantization contract.
INT8 retains supplied G256 FP32 scales, exact BF16 rounding, and scores applied
after BF16 Down. Synthetic accuracy checks do not establish model accuracy.

Raw measurements, profiles, independent audits, rejected-trial sources, frozen
packages, build metadata/binaries, clock observations and a SHA-256 manifest are
outside Git in `/workspace/artifacts/thor_moe_interleaved_mega_20261021`. The main raw table is
`measurement/moe_interleaved_mega_final.json`; the combined-output pilots are
preserved alongside it. The repository contains readable code/reports, not raw
logs or JSON result dumps.
