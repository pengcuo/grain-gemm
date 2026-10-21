# Thor: Mega BF16 versus Mega INT8, Mega FP8 and Cosmos BF16

DeepInfer's BF16 Mega implementation runs correctly on Thor after extending
its host architecture dispatch to exact SM110. Its complete FFN is
**1.51–1.77x faster** than Cosmos BF16 in this four-way run, reducing latency
by **33.8–43.4%**. Mega INT8 and Mega FP8 remain faster at every measured size.

## Same-run complete FFN measurements

GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**;
EMC was **locked at 2750 MHz**. E=32, H=1536, I=1024, Top-4, SwiGLU, no shared
experts, one Thor GPU and no communication. All four columns below were measured
again in this run; no historical BF16 values are mixed into the table.

| Tokens | Mega INT8 ms (TOPS) | DeepInfer Mega FP8 ms (TFLOPS) | DeepInfer Mega BF16 ms (TFLOPS) | Cosmos BF16 ms (TFLOPS) | Cosmos / Mega BF16 time |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.188 (48.50) | **2.367 (65.31)** | 3.787 (40.82) | 5.724 (27.01) | 1.51x |
| 8192 | 4.608 (67.11) | **3.938 (78.53)** | 5.824 (53.10) | 9.196 (33.63) | 1.58x |
| 16384 | 8.409 (73.55) | **6.946 (89.04)** | 9.943 (62.20) | 16.745 (36.94) | 1.68x |
| 32768 | 16.263 (76.06) | **13.099 (94.43)** | 18.415 (67.17) | 32.510 (38.05) | 1.77x |

Throughput uses useful operations for **Gate, Up and Down together**:
`6*T*4*1536*1024`. Padding and non-GEMM arithmetic are excluded from the
numerator. INT8 is reported in TOPS; floating-point methods are in TFLOPS.
These are complete FFN rates, not isolated GEMM rates. At 32768 tokens,
Mega INT8 is 1.13x faster than Mega BF16 and Mega FP8 is 1.41x faster.

Each result is the median of four round medians, with three samples per method
per round and two warm calls. Method order rotates and reverses so each method
occupies each position equally. A common preallocated 256 MB flush precedes
each sample outside its CUDA-event interval. The timer covers the eager call,
including host submission gaps; full-call CUDA Graphs are not used.
No archive copying overlapped GPU timing.

## Compared implementations and timing scope

- **Mega INT8:** the selected interleaved G256 experiment, CW16 and exact SiLU
  table. Public validation, routing, capacity read, per-call allocations, fused
  Gate/Up–SwiGLU–quantization–Down and separate weighted combination are timed.
  Hidden INT8 values/scales stay on chip. This is the existing selected
  [experimental implementation](../../../experiments/thor_interleaved_mega/README.md).
- **Mega FP8:** the existing DeepInfer/DeepGEMM FP8 E4M3 G32 UE8M0 implementation.
  Input is already quantized in its reusable buffer. Route ID/score copies,
  output allocation and the complete Mega kernel are timed.
- **Mega BF16:** `deep_gemm.bf16_mega_moe`, BF16 inputs and weights. Each timed
  call copies BF16 input, route IDs and scores into its reusable buffer,
  allocates output, and executes the complete Mega kernel. Gate/Up, SwiGLU,
  Down and final combination are fused into the persistent execution, but
  hidden BF16 ring buffers and BF16 output slots still use global memory.
- **Cosmos BF16:** the original Cosmos routing/counts, stable sorting, padding,
  gather, two `torch._grouped_mm` projections, SwiGLU/score multiplication and
  BF16 scatter-add, compiled using default
  `torch.compile(fullgraph=True, dynamic=True)`.

All methods share the original BF16 input/weights, route IDs and normalized
scores rounded through BF16. Router logits/top-k, compilation, one-time weight
transforms/quantization and reusable Mega workspace setup are outside timing.
INT8 and FP8 entry quantization is also outside timing; the BF16 paths receive
the original BF16 activations. Thus this compares the existing APIs under the
previous prequantized-input policy, not BF16-to-low-precision conversion plus FFN.

Both DeepGEMM Mega paths use `activation_clamp=None` and `fast_math=True`.
Their route scores multiply the hidden activation before Down. The INT8 path
preserves grain-gemm's score-after-BF16-Down behavior. These operations agree
in real arithmetic but differ in rounding, so cross-implementation bit identity
is not required. The BF16 Mega baseline is substantially stronger than the
Cosmos baseline; the earlier Cosmos numbers do not represent a BF16 ceiling.

## Correctness and clocks

All four shapes passed before and after timing. BF16 results were checked over
the **entire output**, using an independent per-expert `torch.mm` reference
with BF16 projection outputs, FP32 SwiGLU/score multiplication, one hidden BF16
rounding before Down, and FP32 slot-order combination. This reference does not
call grouped GEMM or Mega. Two-token CPU references provide a separate check.

Mega BF16's observed full-output relative L2 error was **0.0058–0.0066%**,
with maximum absolute error **0.0078125** against that reference. Cosmos BF16's
relative L2 error was approximately 0.296%. Its original GEMM, pointwise and
scatter evaluation need not produce the same bits as this reference.
Full-output checks use explicit bounds
`rtol=0.025, atol=0.008`, relative L2 at most 0.03 and maximum absolute error
at most 0.05. These are synthetic numerical checks, not model quality results.

The first BF16 Mega call matched every copied input/route/score bit and all
expert counts. The INT8 experiment matched full hidden INT8 values, FP32 scale
bits, padding, useful expert output and final output bits against the frozen
predecessor. FP8 input values/scales matched the original cast. All four methods
also repeated the same complete output bits in the observed repeat checks.

The run recorded 192 timed calls and checked GPU and EMC min/max/current clocks
at all 384 sample endpoints. All 1563 recorded clock/temperature observations
matched the requested frequencies. Cooling starts at 62 C and resumes at 54 C;
the abort threshold is 70 C. Maximum sampled temperature was **63.906 C**.
Source, native library and payload hashes were unchanged across the run.

## Adaptation and reproducibility

The repository now includes a [portable builder and comparison harness](../../../experiments/thor_mega_bf16/README.md).
It passed an untimed 4096-token output/reference smoke check. The table above
remains the original four-shape measurement from the separately frozen harness;
porting its dependency paths did not generate replacement timing samples.

DeepInfer is `8b16149741a4935e1dec8cea27853f273b374c5b`; vendored DeepGEMM is
`1fc9c66975ee322d628b3288534ca0e9ce12a64d`. In a separate source/build tree,
the existing FP8 compatibility source gained one additional host condition:
BF16 accepts `arch_major == 11 && arch_minor == 0` as well as the original
major 10. Device arithmetic, scheduling, weight layout and real JIT target are
unchanged. This is a Thor compatibility experiment, not unchanged upstream
architecture support. It does not modify grain-gemm's production dispatcher
or the GB10 implementation.

The isolated DeepGEMM extension SHA256 is
`6da6215fcf6e2e72c6f5cd620fb4fe62ed6827a9df6b038414a7264bca552aa3`;
the selected INT8 library remains
`f54e074b67cf69ff9e75d669d4a5094f75fa221bf40bb200a957851433d8d6fe`.

Raw measurements, the independent CPU audit, smoke results, the exact harness,
adapter, compatibility patch, build metadata and frozen dependencies are kept
outside Git in `/workspace/artifacts/thor_moe_mega_bf16_20261021`. The primary
measurement is `measurement/moe_mega_bf16_comparison_final.json`. The initial setup attempt
was rejected before any test case because packaged JIT headers were missing;
the headers were copied from the pinned dependency, all 919 payload files were
verified, and the successful smoke preceded formal timing. Its rejected log
is retained separately and contributes no timing samples.
