# Thor MoE FFN: Cosmos BF16, current INT8 and DeepInfer FP8 Mega

Fresh same-run comparison after the INT8 per-expert padding-skip update.
GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**;
EMC was locked at **2750 MHz**. GPU/EMC min, max and current readings were
verified throughout the run. No clock settings were changed. The maximum
sampled temperature was **63.0°C**; cooling starts at 62°C and execution aborts
at 70°C.

E=32 experts, H=1536, I=1024, Top-4, SwiGLU, no shared experts, one Thor GPU,
no communication. Every method receives the same original BF16 inputs/weights,
expert IDs and routing-score values rounded through BF16. The first four rows
use random Top-4 routing. The uneven case has 20384 input tokens, with 16 experts
receiving exactly 4096 assignments each and 16 receiving 1000 each.

| Input tokens | Cosmos BF16 ms (TFLOPS) | Current INT8 ms (TOPS) | DeepInfer FP8 Mega ms (TFLOPS) | BF16 / INT8 time |
|---:|---:|---:|---:|---:|
| 4096 | 5.465 (28.3) | 3.291 (47.0) | 2.396 (64.5) | 1.66× |
| 8192 | 9.420 (32.8) | 5.394 (57.3) | 4.185 (73.9) | 1.75× |
| 16384 | 16.523 (37.4) | 9.484 (65.2) | 7.364 (84.0) | 1.74× |
| 32768 | 31.879 (38.8) | 18.496 (66.9) | 13.972 (88.5) | 1.72× |
| 20384 (uneven) | 19.086 (40.3) | 12.969 (59.3) | 8.987 (85.6) | 1.47× |

INT8 is **1.47–1.75× faster** than this compiled Cosmos BF16 implementation in
these five cases. DeepInfer FP8 remains faster than INT8. These are complete
FFN useful-operation rates, rather than the throughput of either standalone
GEMM. The INT8 and FP8 columns are newly measured alongside BF16; the earlier
FP8/INT8-only run is retained in its own report and is not substituted here.

## Cosmos implementation

Official Cosmos Framework revision
`4ed6244506e00bb270ed17d4532897f2aaa7bc16`, executing the unchanged
`Qwen3VLMoeTextExpertsGroupedMm`, `_run_experts_grouped_mm`,
`_weighted_expert_counts` and original Triton padding module. It uses stable
expert sorting, per-expert padding to multiples of 16, gather, **two
`torch._grouped_mm` calls**, SwiGLU and routing-score multiplication before down,
then BF16 scatter-add. Counts are computed inside each timed forward.

Compilation follows the existing Cosmos baseline:
`torch.compile(make_cosmos_forward(module), fullgraph=True, dynamic=True)`,
default Inductor backend/mode, with no extra max-autotune setting. Compilation
is outside timing. The loader extracts unchanged upstream AST definitions to
avoid unrelated framework dependencies; the original kernel module is loaded
from the checkout. Missing Transformers is handled by the same PyTorch
`nn.functional.silu` operation used by its SiLUActivation provider. Full source,
loader, activation and weight-stride metadata are recorded in the raw report.
Original Cosmos weights remain N-contiguous BF16 Parameters, populated once
from the common Linear weight values outside timing. BF16 scores are prepared
outside timing and have the same numeric values as the other methods' scores.

## Complete-call timing

Current INT8 calls the actual public `moe_ffn(..., backend="auto")` including
validation, routing/counting and CPU capacity read, allocation, BF16 input
quantization/packing, both projections, SwiGLU/intermediate quantization and
weighted output combination. At 4096/8192/16384 tokens it selects cached two-SM
fused gate/up plus activation/quantization; the other two sizes retain separate
gate/up and activation. Down and output combination are separate stages.

FP8 includes BF16-to-FP8 input quantization directly into the Mega input buffer,
ID/score copies, output allocation and the Mega kernel's dispatch, both
projections, activation/quantization and combination. Reusable Mega workspace
setup is outside timing. All methods exclude router logits/top-k, original
input generation, one-time weight quantization/layout setup and compilation.
These are complete eager CUDA-event spans including submission/synchronization
gaps inside a call. No full-call CUDA Graph timing is used.

Two short warm calls per method precede **four rounds of three samples**.
Order rotates and reverses across four methods, including a frozen pre-padding
INT8 baseline retained in the raw results, so every method occupies every
position equally at each sample index. All methods use the same preallocated
256 MB L2 flush outside each timing interval. Table values are medians of the
four round medians. No long sustained benchmark warmup is used.

Throughput counts `6*T*4*1536*1024` useful projection operations, including both
gate/up and down, excluding padding and non-GEMM arithmetic. Integer rates are
TOPS; floating-point rates are TFLOPS. Across-method numerical equality is not
required because quantization, score placement and rounding differ.

## Correctness and provenance

All five cases passed:

- Current/pre-padding INT8 full QH values, FP32 scale bits, counts, padding and
  final BF16 output bits match. Valid output/reference checks pass again after
  timing.
- FP8 input values and packed scales match the upstream input cast bitwise;
  first-call expert counts match the routes. Repeated FP8 outputs are bitwise
  stable and pass independent quantized CPU-reference checks.
- Cosmos complete outputs are finite. Sampled output tokens pass an independent
  BF16 CPU reference with fused SwiGLU/score rounding, `rtol=0.025, atol=0.008`,
  before and after timing. Repeated full outputs also pass this tolerance;
  bitwise stability is recorded but not required for BF16 scatter-add.
- Differences against a common unquantized FP32 reference are recorded as a
  synthetic numerical check, not a model-quality result.
- Source, package, script and native-binary fingerprints remained unchanged
  during the run, including the upstream Cosmos source hashes.

DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`, vendored DeepGEMM
`1fc9c66975ee322d628b3288534ca0e9ce12a64d`, with the previously validated three
host SM110 architecture-acceptance guard changes and direct input quantizer.
Device arithmetic/layout/scheduling are unchanged. FP8 uses E4M3 G32 UE8M0,
`fast_math=True`, with scores before intermediate quantization/down. INT8 uses
G256 FP32 scales and scores after BF16-rounded down outputs. Cosmos uses BF16
weights/activations and scores before down with BF16 scatter-add.

Raw samples, CSV, clock/temperature snapshots, correctness/provenance, harnesses,
frozen INT8 packages and the five Cosmos source files are outside Git at
`/workspace/artifacts/thor_moe_cosmos_fp8_int8_retest_20261021/`.

Related: [preceding INT8/FP8 retest](moe_ffn_padding_skip_vs_deepinfer_emc2750.md),
[padding-skip behavior](moe_ffn_padding_skip_emc2750.md),
[original Cosmos baseline methodology](moe_ffn_cosmos.md).
