# Thor INT8 MoE: skip wholly padded expert tiles

GPU frequency was reduced to the nominal **1.2 GHz** setting (actual locked
**1197 MHz**); EMC was locked to **2750 MHz**. All six GPU/EMC min, max and
current frequency readings were checked during timing. GPU temperatures during
the public-call measurements were 51.4–60.8°C. No clock settings were changed.

## Behavior

Each expert uses its own GPU INT32 token count. Before entering GEMM work,
the scheduler skips a cluster if its first row is outside that expert's valid
prefix. This includes the first scheduled task and empty experts. A persistent
worker that has no live tasks returns before pipeline setup; other workers
continue at their next live task. All roles and both CTAs of a two-SM cluster
follow the same decision, preserving the collective's synchronization.

Skipped clusters perform no GEMM input/weight loads, matrix arithmetic or
output writes. For a 1000-row expert in a 4096-row allocation, both current
128-row and 256-row families execute through row 1023 and skip rows 1024–4095:
**75% of its allocated GEMM rows are skipped**. The last partially live cluster
still computes its full tile. Counts are read again on CUDA Graph replay.

The public shape/stride/scale layouts and G256 numerical contract are unchanged.
Public Thor MoE dispatch now uses counted configurations 43/44/47; configuration
46 exposes another counted geometry. The fused gate/up path already skips
wholly padded compute clusters and retains its dispatch. Original uncounted
configurations 34/36/45 remain available for isolated comparisons. Single-GEMM,
shared CUTLASS overrides and SM120/GB10 implementations were unchanged by this
padding-skip update.

Capacity-based buffers are still allocated. Input/intermediate padding retains
its required zero initialization, and the fused path still writes its padded
INT8/scale values. This change reduces GEMM padding work; it does not eliminate
all padding IO. Traffic savings here follow from skipped code paths, not DRAM
hardware-counter measurements.

## Complete public API measurements

E=32, H=1536, I=1024, Top-4, BF16 input/output, G256 INT8 weights.
The comparison uses the actual `moe_ffn(..., backend="auto")` calls before and
after this update, including routing, allocation, quantization, both projections,
activation and combination. The before package was frozen immediately before
this update and already includes the cached two-SM fused gate/up implementation.
Every reported final output was bitwise identical between the two packages.

Each result is the median of eight complete eager-call CUDA-event samples:
four rounds of two samples with reversed paired implementation order. A 256 MB
cache flush runs outside the timing interval before each call. These are eager
GPU event spans, including gaps inside a call; they are not isolated GEMM or
CUDA-graph FFN timings. Throughput counts useful assignments only:
`6 * T * 4 * 1536 * 1024 / time`, reported as INT8 TOPS.

| Input tokens | Routing | Capacity per expert | Before ms | After ms | After useful TOPS | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | Random Top-4 | 640 | 3.103 | 3.100 | 49.88 | 1.001× |
| 8192 | Random Top-4 | 1152 | 5.165 | 5.167 | 59.85 | 1.000× |
| 16384 | Random Top-4 | 2112 | 9.547 | 9.531 | 64.89 | 1.002× |
| 32768 | Random Top-4 | 4288 | 18.410 | 18.442 | 67.07 | 0.998× |
| 20384 | 16 experts × 4096 + 16 × 1000 | 4096 | 16.069 | 12.967 | 59.34 | 1.239× |

The uneven case has 81536 assignments: 16 experts receive 4096 rows each and
16 receive 1000 rows each. Equal-capacity storage reserves 131072 rows, so this
case remains within the native auto-dispatch padding threshold. Whole-FFN
latency falls **19.30%**, or **1.239×** speedup. Random-routing results remain
within 0.2% of the frozen baseline in this run; there is little fully unused
padding in those cases, and several already use fused gate/up.

## Isolated projection check

This separate experiment starts from packed/quantized operands at E32, C4096,
H1536, I1024. Gate/up includes both projections, N=2048; down has N=1536.
It compares original dynamic configuration 45 against counted configuration 47
with the same 256×256×256 geometry. Results use isolated two-launch CUDA graphs,
three rounds of two samples and a cache flush before each sample. This timing
scope differs from the complete eager public calls above.

| Expert row counts | Gate/up before → after ms | Down before → after ms |
|---|---:|---:|
| All 4096 | 5.454 → 5.398 | 3.609 → 3.528 |
| 16 × 4096, 16 × 1000 | 5.460 → 3.647 | 3.605 → 2.333 |
| 1 × 4096, 31 × 1000 | 5.467 → 2.016 | 3.607 → 1.211 |
| 4096, 1000, 30 empty | 5.613 → 0.497 | 3.603 → 0.343 |

The last two rows test projection behavior directly; their severe skew would
make the public auto FFN choose the ragged Triton fallback. They are not claims
of public native FFN latency. Valid projection output rows were bitwise equal.

## Validation and evidence

- Production MoE regression suite: **198 passed, 2 skipped** across
  `test_moe_ffn.py`, `test_moe_fused.py`, `test_moe_fused_sm110.py`,
  `test_moe_input_io.py` and `test_moe_routes_sm110.py`.
- The counted-scheduler tests poison output with NaNs and assert that wholly
  padded clusters remain unwritten, including initial work and empty experts.
  GPU counts change between graph replays; valid rows are compared against an
  independent CPU grouped-dot reference. FFN poison tests check that unwritten
  gate/up padding does not leak into activation or the final result.
- The isolated candidate passed 288 combinations spanning counts, capacities,
  tile configurations, raster orders, swizzles, eager and replay execution.
  Valid rows match original output bits; whole padding clusters remain NaN;
  omitting counts still computes the full allocation.
- Compute Sanitizer memcheck on that same candidate algorithm reported **0
  errors**. The production build uses the same algorithm with its original C ABI.
- Before/after source and native-library SHA-256 fingerprints were checked for
  stability throughout the public-call benchmark.

Raw samples, clock/temperature readings, test/build/sanitizer logs, frozen
packages and harnesses are outside Git at
`/workspace/artifacts/thor_moe_padding_skip_20261021/`, with a SHA-256 manifest.
Build the production library with:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /workspace/cutlass
```
