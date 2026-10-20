# Thor MoE native tile selection at 1.2 GHz

Measured on 2026-10-20. **GPU frequency was reduced to nominal 1.2 GHz,
with the existing lock and observed frequency at 1197 MHz.**

The 128-row config 34 loses throughput to the original Cosmos 256-row tile
on larger expert matrices. The independent SM110 MoE library now includes
config 45: a dynamic persistent 2SM 256x256x256 tile with the existing G256
mainloop, BF16 epilogue and GrainGEMM scale layouts. Each projection selects
its tile independently at measured capacities. The public API, numerical
rounding points, single-GEMM implementations and GB10 dispatch are unchanged.
The capacity lists are documented in [the MoE API](../../../docs/moe_ffn.md).

## Conditions and timing

- NVIDIA Thor SM110, 20 SMs; CUDA/NVCC 13.2, PyTorch
  2.11.0a0+a6c236b9fd.nv26.03.46836102 and Triton 3.6.0.
- E=32, H=1536, I=1024, G256. Expert GEMMs consume INT8 operands and
  FP32 scales and produce BF16. Full FFN uses BF16 inputs and top-k=4.
- CUTLASS `098de2a652cf8f00fd70b2df54051c7eccbb855a`; original Cosmos
  source `8977529aa23f70f02bfc36559599f00d59792996`, rebuilt with the same
  current toolchain. These are fresh measurements, separate from its
  historical 1386 MHz report.
- EMC policy remains dynamic, 665.6–4266 MHz. Timing endpoint snapshots
  were 4266 MHz; these observations do not establish a continuous memory
  clock lock. No thermal slowdown was observed. Maximum sampled temperatures
  were 74.28 C for the capacity sweep and 64.44 C for full FFN.

GEMM-only timing uses preallocated operands, repeated weights without an L2
flush, at least 200 ms of GPU warmup per method, rotating method order and
three rounds of CUDA Graph timing. Each round has ten replay samples of
10 ms captured batches; the reported statistic is the median of round
medians. The separate capacity-2112 supplement uses five rounds. Original
Cosmos activation scales are converted outside timing so both implementations
consume identical numerical inputs. Routing, quantization and activation are
outside these GEMM timers.

## GEMM-only recovery

Gate/up uses N=2048, K=1536; down uses N=1536, K=1024. TOPS is
`2*E*M*N*K / (seconds*1e12)`, without counting extra internal tile padding.
Here **M is the row count per expert**, and all M rows enter the numerator.

| M per expert | Projection | Original Cosmos20 TOPS | Previous config34 TOPS | Config45 TOPS |
|---:|---|---:|---:|---:|
| 1024 | Gate/up | 164.86 | 136.99 | 164.10 |
| 1024 | Down | 132.74 | 128.73 | 132.77 |
| 4096 | Gate/up | 178.24 | 159.19 | 177.96 |
| 4096 | Down | 150.40 | 142.60 | 151.55 |

The sweep covers 28 capacity/projection pairs, followed by two pairs at
capacity 2112. Config45 is selected only for the 22 measured pairs with
more than 2% median throughput improvement. Capacity 640 and 896 retain
config34. At capacity 1152, gate/up throughput improves by 11.0%, while down
latency increases by 3.55%, so auto selects 45/34. Down at 1088, 2112 and 2176 retains
34 because its measured improvement is below 2%. Uncalibrated shapes keep
their previous selection.

## Complete FFN

Before is the frozen package from `ee3a581d9c4a0acb3435b8f9ba99a0e5ca0f9513`;
after uses the new automatic tile selection. Each paired case shares its
inputs, weights, scales and routing. The timer covers the complete eager
public call, including route preparation, quantization/packing, both GEMMs,
SwiGLU/requantization, combination and host dispatch gaps. Weight preparation,
router/top-k selection, compilation, correctness checks and warmup are outside
timing. Five rotating rounds with three samples per method per round produce
the median of round medians. No CUDA Graph is used for these full-call timers.

Effective TOPS uses `6*T*top_k*H*I / (seconds*1e12)`: both projections are
included, padding is excluded, and non-GEMM operations add time without adding
to the numerator. C is the actual common packed capacity per expert.

| Input tokens T | Routing | C | New configs gate/up, down | Before ms | After ms | Before effective TOPS | After effective TOPS | Speedup |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 4096 | Random | 640 | 34, 34 | 2.4990 | 2.4987 | 61.87 | 61.88 | 1.000x |
| 8192 | Balanced | 1024 | 45, 45 | 4.1400 | 3.8705 | 74.70 | 79.90 | 1.070x |
| 8192 | Random | 1152 | 45, 34 | 4.4394 | 4.2868 | 69.66 | 72.14 | 1.036x |
| 16384 | Random | 2112 | 45, 34 | 7.9982 | 7.7930 | 77.33 | 79.36 | 1.026x |
| 32768 | Balanced | 4096 | 45, 45 | 14.9991 | 14.3036 | 82.47 | 86.48 | 1.049x |
| 32768 | Random | 4288 | 45, 45 | 15.5367 | 14.8072 | 79.61 | 83.54 | 1.049x |

The 32768-token balanced case has M=4096 per expert. The 4096-token case has
512 effective rows per expert on average and C=640. They are different loads.
These paired FFN results preserve the separate
[earlier four-method comparison](moe_ffn_optimized.md); they do not replace it.

## Validation and rebuild

- Complete MoE test file: **126 passed, 2 skipped**.
- Config45 partial-M/N and offset-view tests under memcheck: **0 errors**.
- Across the sweep and supplement, 1920 independent CPU G256-FMA sample
  checks and 90 complete BF16 bit-pattern comparisons passed.
- All six full-FFN cases have identical before/after output bit patterns;
  both implementations match two-token independent CPU references in these
  cases. Source, native-library and benchmark-helper hashes were checked
  before and after timing.

From the repository root, rebuild the separate MoE library and run validation:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /path/to/cutlass
PYTHONPATH=src python -m pytest -q tests/test_moe_ffn.py
PYTHONPATH=src compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python -m pytest -q tests/test_moe_ffn.py -k 'thor_moe_tiles_preserve_group_math_and_offset_views and 45'
```

The explicit CUDA API error mode follows the existing
[container validation setup](README.md#correctness). The measured MoE native
library SHA256 is
`f7596c6efb2c92ff6439905634ff789afecd57e3a8ab20cdbb8ab1846beac5ba`.
