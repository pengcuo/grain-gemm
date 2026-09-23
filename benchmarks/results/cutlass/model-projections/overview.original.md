# GB10: G256 model projections at M = 1024 and 2048

All 18 shapes were measured afresh with the same software and native libraries. Inputs are prequantized INT8; group scales and accumulation are FP32; output is BF16. Every backend is tuned independently: 17 CuTe, 8 CUTLASS, and 9 Triton candidates. The selected configurations are measured again on different input data.

CUDA Graph timing uses five rounds, five replays per round, and the median of round medians. The backend order rotates every round. Inputs are reused without cache flushing and output is preallocated. Timing includes GEMM, group scaling, accumulation, and BF16 output conversion. It excludes input quantization, compilation, tuning, allocation, and host dispatch.

Equivalent TFLOPS = 2MNK / (latency_us × 10^6). Actual INT8 arithmetic throughput is TOPS; the equivalent-TFLOPS figure uses the same numerical operation-count convention.

M is the actual number of GEMM rows. These are individual projection benchmarks, not end-to-end model or MoE measurements. Routed experts have their own row counts; no M=T/8 transformation is applied here. Separate and fused projections are alternatives, not additive costs. LM head rows evaluate all M tokens. The N=32 router is outside the current native backends' shape support.

## M = 1024

| Projection | M | N | K | CuTe µs | CUTLASS µs | Triton µs | Equivalent TFLOPS: CuTe / CUTLASS / Triton |
|---|---:|---:|---:|---:|---:|---:|---|
| FFN gate or up | 1024 | 1024 | 1536 | 28.187 | 29.198 | 33.435 | 114.28 / 110.32 / 96.34 |
| FFN down | 1024 | 1536 | 1024 | 24.101 | 25.276 | 34.513 | 133.65 / 127.44 / 93.33 |
| FFN fused gate + up | 1024 | 2048 | 1536 | 47.156 | 49.302 | 65.383 | 136.62 / 130.67 / 98.53 |
| Q projection | 1024 | 3072 | 1536 | 63.678 | 68.477 | 90.721 | 151.76 / 141.12 / 106.52 |
| K or V projection | 1024 | 512 | 1536 | 15.483 | 16.125 | 19.322 | 104.03 / 99.88 / 83.36 |
| Fused QKV | 1024 | 4096 | 1536 | 94.242 | 98.204 | 121.883 | 136.72 / 131.21 / 105.72 |
| Attention output | 1024 | 1536 | 3072 | 59.252 | 60.737 | 81.105 | 163.10 / 159.11 / 119.15 |
| LM head (151936) | 1024 | 151936 | 1536 | 3299.392 | 3496.096 | 4304.128 | 144.86 / 136.71 / 111.04 |
| LM head (262144) | 1024 | 262144 | 1536 | 5804.096 | 6136.672 | 7411.904 | 142.08 / 134.38 / 111.26 |

## M = 2048

| Projection | M | N | K | CuTe µs | CUTLASS µs | Triton µs | Equivalent TFLOPS: CuTe / CUTLASS / Triton |
|---|---:|---:|---:|---:|---:|---:|---|
| FFN gate or up | 2048 | 1024 | 1536 | 47.177 | 49.500 | 65.718 | 136.56 / 130.15 / 98.03 |
| FFN down | 2048 | 1536 | 1024 | 48.097 | 50.429 | 64.496 | 133.95 / 127.75 / 99.89 |
| FFN fused gate + up | 2048 | 2048 | 1536 | 95.836 | 100.542 | 122.595 | 134.45 / 128.15 / 105.10 |
| Q projection | 2048 | 3072 | 1536 | 139.399 | 143.635 | 178.367 | 138.65 / 134.56 / 108.36 |
| K or V projection | 2048 | 512 | 1536 | 28.240 | 29.270 | 33.277 | 114.07 / 110.05 / 96.80 |
| Fused QKV | 2048 | 4096 | 1536 | 196.102 | 198.284 | 237.611 | 131.41 / 129.96 / 108.45 |
| Attention output | 2048 | 1536 | 3072 | 120.955 | 123.900 | 160.252 | 159.79 / 155.99 / 120.61 |
| LM head (151936) | 2048 | 151936 | 1536 | 6781.088 | 7580.672 | 8700.992 | 140.97 / 126.10 / 109.86 |
| LM head (262144) | 2048 | 262144 | 1536 | 11624.544 | 12807.104 | 15010.720 | 141.88 / 128.78 / 109.87 |

## Reproduction and provenance

Run the local `compare_cutlass.py` with `--output-dir results`; its adjacent `tune_grain.py` defines the nine Triton candidates used here. `scope.json` lists all shapes. `measurement_sources/` preserves the package and benchmark source used for this run. The case JSON files contain source/library hashes, seeds, sampled correctness checks, every candidate's timing, and all final rounds.

Correctness uses independent CPU INT32 group dot products with FP32 FMA rounding at 32×32 sampled positions, full-output finite checks, and zero/restore graph replay for every selected kernel. This is sampled numerical validation, not exhaustive validation of every output element.

[Raw results and timing ranges](results/report.md) · [CSV](results/results.csv)

These measurements do not update the library's automatic dispatch table. Results characterize these implementations, candidate sets, and this GB10 environment; small gaps should be read alongside the raw timing ranges.
