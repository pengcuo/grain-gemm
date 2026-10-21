# Thor K1536/N2048 GEMM: EMC2750 versus EMC4266

Measured 2026-10-21 19:57 CST. GPU remains locked at **1197 MHz (reduced to approximately 1.2 GHz)**. EMC is locked at 2750 MHz and then 4266 MHz in sequential runs. After testing, EMC was restored to the dynamic **665.6–4266 MHz** range; GPU limits were unchanged.

Only N changes relative to the [N1024 EMC comparison](gemm_k1536_n1024_emc_comparison.md): now `K=1536, N=2048`, with M1024/2048/4096/8192. CuTe uses the same fixed independent CUDA **config #27**, not a fresh shape-specific tuning. Torch BF16 uses default `torch.mm`. All outputs are BF16.

## Repeated-input results

| M | INT8 at EMC2750 TOPS | INT8 at EMC4266 TOPS | BF16 at EMC2750 TFLOPS | BF16 at EMC4266 TFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 107.95 | 116.91 | 101.68 | 108.83 |
| 2048 | 137.14 | 145.47 | 132.98 | 139.18 |
| 4096 | 160.01 | 164.38 | 129.14 | 160.64 |
| 8192 | 139.48 | 159.40 | 130.74 | 152.64 |

No tested case exceeds 200 TOPS. The highest measured CuTe value is **164.38 TOPS** at M4096/EMC4266; BF16 at this shape reaches **160.64 TFLOPS**. These values describe fixed config #27 and default torch.mm under this timing protocol, not an exhaustive performance limit.

## Latencies and cold-cache results

| EMC MHz | M | INT8 cold µs (TOPS) | INT8 warm µs (TOPS) | BF16 cold µs (TFLOPS) | BF16 warm µs (TFLOPS) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2750 | 1024 | 124.000 (51.96) | 59.680 (107.95) | 161.184 (39.97) | 63.360 (101.68) |
| 2750 | 2048 | 161.984 (79.54) | 93.952 (137.14) | 212.768 (60.56) | 96.896 (132.98) |
| 2750 | 4096 | 248.000 (103.91) | 161.056 (160.01) | 303.040 (85.04) | 199.552 (129.14) |
| 2750 | 8192 | 421.344 (122.32) | 369.504 (139.48) | 485.856 (106.08) | 394.208 (130.74) |
| 4266 | 1024 | 83.264 (77.37) | 55.104 (116.91) | 109.952 (58.59) | 59.200 (108.83) |
| 4266 | 2048 | 120.032 (107.35) | 88.576 (145.47) | 152.160 (84.68) | 92.576 (139.18) |
| 4266 | 4096 | 196.128 (131.39) | 156.768 (164.38) | 225.600 (114.23) | 160.416 (160.64) |
| 4266 | 8192 | 351.456 (146.65) | 323.328 (159.40) | 405.408 (127.13) | 337.664 (152.64) |

## Method and validation

A is contiguous `[M,1536]`; B is the transpose view of contiguous Linear weights `[2048,1536]`. INT8 G256 operands are quantized from the same original BF16 inputs/weights before timing. FP32 scales are contiguous `[M,6]` and `[6,2048]`. BF16 uses `torch.mm(x, weight.T, out=out)` without changing backend or precision flags. Outputs are preallocated. Routing, packing, quantization, activation, combine and host dispatch are excluded.

Each graph replay contains one GEMM. Each method/cache mode uses five rounds of five samples, reporting the median of round medians. Cold samples clear a 256 MB buffer outside the timed events and alternate method order; warm samples execute each method in its own repeated-input block without cache flush or interference from the other method. There is no sustained warmup.

Both runs passed sampled bit-exact INT8-output checks against independent CPU INT32 G256 dot products with FP32 scaled accumulation. BF16 results passed sampled CPU FP64-reference comparisons (`rtol=1/128`, `atol=0.05`). Each comparison samples 27 elements across three rows and nine columns, before and after timing. Every output element was finite. These are sampled numerical comparisons, not exhaustive output comparisons.

All GPU/EMC min/max/current readings matched their respective locks before and after timed samples. Source and native binary hashes were stable during each run and identical across the two runs. No production kernel or dispatch changes were made.

| EMC MHz | Maximum GPU temperature |
| ---: | ---: |
| 2750 | 58.406 °C |
| 4266 | 58.812 °C |

The guard pauses at 62 °C until 54 °C and aborts at 70 °C. The benchmark reads clock state but does not change it.

## Reproduction

Set the requested EMC lock on the host while retaining the GPU1197 MHz lock, then run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --n 2048 --emc-hz 2750000000 --output /tmp/thor_gemm_n2048_emc2750.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --n 2048 --emc-hz 4266000000 --output /tmp/thor_gemm_n2048_emc4266.json
```

Raw JSON, logs and measured source/native snapshots are archived outside Git in `/workspace/artifacts/thor_gemm_n2048_emc_20261021/`.
