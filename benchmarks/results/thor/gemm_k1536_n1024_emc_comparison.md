# Thor K1536/N1024 GEMM: EMC frequency comparison

Measured 2026-10-21 19:54 CST. **GPU remains locked at 1197 MHz (reduced to approximately 1.2 GHz)** in every run. EMC was tested locked at 2750 MHz, dynamically allowed between 665.6 and 4266 MHz, and temporarily locked at 4266 MHz. After the final run, EMC was restored to the dynamic 665.6–4266 MHz range; GPU limits were unchanged.

Dynamic release did not produce a higher observed clock during warm measurements: all before/after warm-sample readings were **2750 MHz**. A separate 4266 MHz lock therefore tests the effect of actually running at the highest available EMC setting. Before/after readings are not continuous in-kernel telemetry.

## Repeated-input results

The following values are fresh, matched-protocol measurements for all three policies, not copies of the preceding separately timed tables. INT8 units are **TOPS** and BF16 units are **TFLOPS**. All methods use BF16 output.

| M | INT8 at EMC2750 TOPS | INT8 dynamic TOPS | INT8 at EMC4266 TOPS | BF16 at EMC2750 TFLOPS | BF16 dynamic TFLOPS | BF16 at EMC4266 TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 97.45 | 86.18 | 90.44 | 113.10 | 101.07 | 99.47 |
| 2048 | 116.44 | 109.54 | 111.29 | 118.57 | 107.78 | 111.35 |
| 4096 | 144.68 | 138.80 | 140.20 | 140.10 | 132.50 | 136.45 |
| 8192 | 160.96 | 156.40 | 160.55 | 96.63 | 94.21 | 138.20 |

For M8192, raising locked EMC from 2750 to 4266 MHz improves BF16 throughput by approximately **43.0%**, while CuTe INT8 throughput is essentially unchanged. Small-shape warm results show no improvement. These observations do not establish which hardware resource limits CuTe; profiling counters would be needed for that conclusion.

## Latency and cold-cache results

Cold samples clear a 256 MB buffer outside each timed replay. Warm samples execute each method in a separate repeated-input block, so the other method does not evict its cache contents between samples. Cold method order rotates and reverses across rounds/samples.

| EMC policy | M | INT8 cold µs (TOPS) | INT8 warm µs (TOPS) | BF16 cold µs (TFLOPS) | BF16 warm µs (TFLOPS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2750 | 1024 | 91.840 (35.07) | 33.056 (97.45) | 114.368 (28.17) | 28.480 (113.10) |
| 2750 | 2048 | 122.176 (52.73) | 55.328 (116.44) | 159.872 (40.30) | 54.336 (118.57) |
| 2750 | 4096 | 174.304 (73.92) | 89.056 (144.68) | 220.768 (58.36) | 91.968 (140.10) |
| 2750 | 8192 | 274.560 (93.86) | 160.096 (160.96) | 336.160 (76.66) | 266.688 (96.63) |
| dynamic | 1024 | 95.104 (33.87) | 37.376 (86.18) | 116.480 (27.65) | 31.872 (101.07) |
| dynamic | 2048 | 124.192 (51.87) | 58.816 (109.54) | 159.776 (40.32) | 59.776 (107.78) |
| dynamic | 4096 | 179.712 (71.70) | 92.832 (138.80) | 224.896 (57.29) | 97.248 (132.50) |
| dynamic | 8192 | 282.112 (91.35) | 164.768 (156.40) | 342.784 (75.18) | 273.536 (94.21) |
| 4266 | 1024 | 67.648 (47.62) | 35.616 (90.44) | 78.944 (40.80) | 32.384 (99.47) |
| 4266 | 2048 | 91.584 (70.34) | 57.888 (111.29) | 111.360 (57.85) | 57.856 (111.35) |
| 4266 | 4096 | 129.152 (99.77) | 91.904 (140.20) | 148.768 (86.61) | 94.432 (136.45) |
| 4266 | 8192 | 213.504 (120.70) | 160.512 (160.55) | 232.928 (110.63) | 186.464 (138.20) |

## Scope, method and validation

Each operation is a single mathematical GEMM with `K=1536, N=1024`. CuTe uses the independent CUDA backend, fixed **config #27**, G256 INT8 operands with contiguous FP32 activation/weight scales, and BF16 output. BF16 uses default `torch.mm(x, weight.T, out=out)`, with BF16 input and output. A is row-major; B is a transpose view of contiguous Linear weights `[N,K]`. INT8 operands are quantized from the same original BF16 tensors before timing. No quantization, routing, packing, activation or combine is timed.

Each graph replay contains one GEMM. Each policy uses five rounds of five samples per method/cache mode. Values are medians of round medians; all individual samples and clock readings are retained. There is no sustained warmup. Policies were measured sequentially, not interleaved. The same fixed CuTe configuration and native binaries are used across policies; there is no policy-specific kernel tuning.

All three runs passed sampled INT8 bit-exact BF16 checks against independent CPU INT32 G256 dot products and FP32 scaled accumulation. BF16 outputs were compared with CPU FP64 references from the original BF16 inputs (`rtol=1/128`, `atol=0.05`). Each check samples 27 elements across three rows and nine columns, before and after timing. All output elements were finite. These are sampled numerical comparisons, not exhaustive numerical comparisons.

Benchmark sources/native fingerprints were stable during each run, and all package/native fingerprints match across policies. After the first two runs, only the benchmark guard and CLI were extended to accept a fixed EMC value of 4266 MHz; measured kernel invocations and timing logic were unchanged.

| EMC policy | Observed EMC frequencies across all recorded states | Maximum GPU temperature |
| --- | --- | ---: |
| 2750 | 2750.0 MHz | 55.125 °C |
| dynamic | 665.6, 2750.0 MHz | 54.906 °C |
| 4266 | 4266.0 MHz | 55.093 °C |

GPU min/max/current readings remained 1197 MHz in all samples. Fixed-policy EMC min/max/current readings all matched the requested clock. The guard pauses at 62 °C until 54 °C and aborts at 70 °C. Benchmark code reads clock state but does not modify it.

## Reproduction

Set the desired EMC policy on the host first while retaining the GPU lock. Then run the corresponding command; the benchmark validates the policy before sampling.

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --output /tmp/thor_gemm_emc2750_paired.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --emc-dynamic --output /tmp/thor_gemm_emc_dynamic_paired.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --emc-hz 4266000000 --output /tmp/thor_gemm_emc4266_paired.json
```

Raw measurements/logs and per-run source/native snapshots are archived outside Git at `/workspace/artifacts/thor_gemm_emc_20261021/`.
