# Thor INT8 MoE FFN optimization

**GPU compute frequency reduced to 1.2 GHz; actual supported locked setting: 1197 MHz.**

NVIDIA Thor SM110, 20 SMs. E=32, H=1536, I=1024, top-k=4, G256, BF16 input/output, SwiGLU, seed 2026. All slots are valid in this performance batch. The public FFN API, logical weight strides and scale layouts are unchanged.

This is a fresh, paired four-method run. The previous INT8 package is frozen at `e4c3614f9e01dbb4d40bc349b372f53479f729c1`; its 26 committed source files are verified against Git blobs before loading under a separate module name. Both INT8 paths use identical quantized weights and FP32 scales. The BF16 paths use the corresponding original BF16 weights. All four receive identical tokens, expert ids and numeric routing scores.

## Complete FFN performance

| Tokens | INT8 before ms | INT8 after ms | Before TOPS | After TOPS | Before / after |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.968 | 0.855 | 4.99 | 5.65 | 1.13x |
| 512 | 1.315 | 1.072 | 14.69 | 18.02 | 1.23x |
| 2048 | 2.567 | 1.523 | 30.12 | 50.77 | 1.69x |
| 3072 | 3.272 | 1.937 | 35.44 | 59.87 | 1.69x |
| 4096 | 4.297 | 2.405 | 35.98 | 64.30 | 1.79x |

| Tokens | vLLM BF16 ms | BF16 TFLOPS | vLLM / INT8 | Cosmos compiled BF16 ms | BF16 TFLOPS | Cosmos / INT8 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.295 | 3.73 | 1.52x | 1.710 | 2.82 | 2.00x |
| 512 | 1.407 | 13.74 | 1.31x | 1.917 | 10.08 | 1.79x |
| 2048 | 2.072 | 37.31 | 1.36x | 2.690 | 28.73 | 1.77x |
| 3072 | 2.624 | 44.19 | 1.35x | 3.256 | 35.62 | 1.68x |
| 4096 | 3.286 | 47.06 | 1.37x | 3.914 | 39.50 | 1.63x |

Throughput counts **both** projections: `2*T*R*H*(2I) + 2*T*R*I*H = 6*T*R*H*I`, divided by full-call time. INT8 uses effective TOPS and BF16 uses effective TFLOPS on the same operation-count scale. Padding, activation and routing arithmetic are excluded from the numerator. This is neither an isolated GEMM peak nor serving-engine throughput.

## Implementation and dispatch

- Stable counting-sort routing for E<=64 and T*R<=65536 replaces the general Torch sort/histogram sequence; larger cases retain the general path.
- Multi-row quantization and packing reduce small-program overhead. For Thor native top-k<=8, input quantization runs once per token/group and scatters directly into packed expert inputs, with disjoint padding initialization.
- Packed SwiGLU skips nonlinear/quantization math for entirely padded blocks while writing zero INT8 values and the original finite zero-group scale. Active rows retain FP32 SwiGLU and exact RN group quantization.
- Wider combine tiles amortize routing-index loads and retain post-down FP32 weighting/reduction.
- A separate Thor MoE library reuses the original Cosmos-derived G256 TMA/TMEM collective with smaller tiles and persistent scheduling. Counts-aware variants skip subsequent whole padding clusters; their initial tile remains to preserve the collective pipeline. Each invocation owns its stream/workspace arguments.

| Tokens | Actual auto path | Packed capacity C | Gate/up config | Down config |
|---:|---|---:|---:|---:|
| 128 | Ragged Triton | — | — | — |
| 512 | Ragged Triton | — | — | — |
| 2048 | Thor CUTLASS MoE | 320 | 44 | 44 |
| 3072 | Thor CUTLASS MoE | 448 | 34 | 34 |
| 4096 | Thor CUTLASS MoE | 640 | 34 | 34 |

Config 34 is a dynamic persistent 2SM 128x256x256 tile. Config 44 is a counts-aware static 1SM 128x128x256 tile. Full-FFN comparisons favor 44 for the two measured E32/C320 projection shapes; the larger capacities favor 34 despite its padding. Other shapes default to 34 in the new library. Missing/stale new builds retain the original native config-32 fallback when available. Auto retains Triton for small or highly skewed routes; captured auto uses Triton. No single-GEMM source, existing configuration table or SM12x/GB10 kernel was changed.

## Protocol and BF16 provenance

Five rounds, five samples per method per round, rotating method order. Each method is warmed for at least two calls and 200 ms. CUDA events surround the full eager Python FFN invocation, including host dispatch gaps and the public INT8 routing-statistics synchronization; no CUDA Graph capture or cache flush is used. Reported latency is the median of the five per-round medians. Compilation, correctness checks, cooling, weight quantization/layout preparation and router/top-k selection are outside timing. Raw samples and measurement order are retained.

- vLLM: original unquantized Triton functional expert FFN and original CUDA alignment/activation/reduction sources at `4e304b4d1ed0037514073305d839a79586d0e9de`. This checkout has no Thor E32/N1024 tuned configuration file, so its original default configuration is used. No new BF16 tuning is claimed.
- Omni: adapter provenance at `da572cddb38383deae9caef33b8d490e292ef071`; the measurement executes its upstream vLLM expert core rather than the full Omni engine, router or distributed runner.
- Cosmos Framework: original BF16 grouped-MM expert path at `4ed6244506e00bb270ed17d4532897f2aaa7bc16`, compiled with `torch.compile(fullgraph=True, dynamic=True)`, default Inductor mode.
- Native INT8: the original Cosmos-derived collective and pinned CUTLASS `098de2a652cf8f00fd70b2df54051c7eccbb855a`; each native binary, build manifest, production source and measurement/build helper is hashed in the JSON.

Routing scores are normalized in FP32 once and rounded to BF16. Cosmos receives those BF16 values, while INT8/vLLM receive their exact FP32 expansion. The existing precision contracts remain: INT8 weights are G256 quantized and its route weighting follows the rounded down projection; vLLM weights the FC2 accumulator before BF16 conversion; Cosmos weights the hidden activation before down projection. These differences are recorded separately from kernel correctness.

Software: PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102, Triton 3.6.0, CUDA 13.2, Python 3.12.3. All recorded round GPU clocks are 1197 MHz. Round temperatures range from 52.50 to 60.91 C, with no recorded thermal slowdown. EMC policy remains dynamic (665.6–4266 MHz); every round-end EMC snapshot is 4266 MHz. EMC was not locked or otherwise changed.

## Correctness and validation

- Full repository suite: **1064 passed, 23 skipped**.
- Compute Sanitizer 2026.1.0: **32 selected MoE cases pass memcheck with 0 errors; racecheck reports 0 errors and 0 warnings**. Coverage includes all nine new native configurations, operand/scale offsets, counts-aware changing-population graphs, complete FFN NaN padding and fused input packing.
- Direct native tests compare independent CPU INT32 group dots and scaled FP32 FMA against BF16 bits, including partial tiles, signed/zero/cancelling scales, INT8 extrema and four-byte scale/count offsets.
- Fused quantization/packing is bitwise equal to the unfused path and independent per-route CPU packing for BF16/FP16, duplicate routes, empty experts, masked NaNs and offset views. Routing tests include stable-order boundaries and general-path fallback.
- Every timed case checks full-output finiteness, both INT8 paths against independent CPU references for two tokens/eight selected routes, and original BF16 references. All sampled INT8 reference outputs are exactly equal after BF16 conversion in this run.
- Complete before/after outputs pass the existing numerical tolerances. Their largest absolute difference is 0.001953125; wider FP32 slot reductions can change addition association. The largest synthetic INT8-vLLM relative L2 difference is 2.2721%. This quantization/rounding difference does not establish model accuracy.

All five cases verify that source, native binary/build and helper hashes remain unchanged during measurement. Results establish these tested shapes and protocols, without a universal optimum claim. Earlier reports remain historical records; no speedup above mixes separate runs.

## Reproduce

Use an unmodified baseline checkout and rebuild its native library with the same toolchain, then build the current Thor libraries. External baseline checkouts must match the revisions above.

```bash
git worktree add --detach /tmp/grain-moe-before e4c3614f9e01dbb4d40bc349b372f53479f729c1
python /tmp/grain-moe-before/tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python benchmarks/bench_moe_optimized.py \
  --before-package-dir /tmp/grain-moe-before/src/grain_gemm \
  --vllm-dir /path/to/vllm --omni-dir /path/to/vllm-omni \
  --cosmos-dir /path/to/cosmos-framework \
  --output benchmarks/results/thor/moe_ffn_optimized.json
```

The benchmark records clock policy and does not set frequencies. Match the reduced 1197 MHz GPU setting before reproducing these numbers.

```bash
PYTHONPATH=src compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python -m pytest -q tests/test_moe_ffn.py -k 'thor_moe_tiles or thor_moe_counted or thor_counted_ffn or quantize_pack'
PYTHONPATH=src compute-sanitizer --tool racecheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python -m pytest -q tests/test_moe_ffn.py -k 'thor_moe_tiles or thor_moe_counted or thor_counted_ffn or quantize_pack'
```

[Raw JSON](moe_ffn_optimized.json) · [CSV with all timing samples](moe_ffn_optimized.csv) · [Public API and layouts](../../../docs/moe_ffn.md)
