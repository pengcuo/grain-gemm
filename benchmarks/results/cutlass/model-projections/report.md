# GB10 G256: CUTLASS collective vs direct CuTe vs Triton

Pure prequantized INT8 GEMM, FP32 group scales and accumulation, BF16 output. Each backend is tuned separately on the same input; selected candidates are timed again on fresh input. CUDA graphs use warm caches and preallocated output. Compilation, tuning, quantization, allocation and host dispatch are excluded.

Latency is μs. Equivalent TFLOPS uses 2MNK/time; the actual INT8 throughput unit is TOPS.

| M | N | K | CUTLASS μs | CuTe μs | Triton μs | CUTLASS / CuTe speedup | Equivalent TFLOPS: CUTLASS / CuTe / Triton |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 512 | 1536 | 16.125 | 15.483 | 19.322 | 0.960× | 99.88 / 104.03 / 83.36 |
| 1024 | 1024 | 1536 | 29.198 | 28.187 | 33.435 | 0.965× | 110.32 / 114.28 / 96.34 |
| 1024 | 1536 | 1024 | 25.276 | 24.101 | 34.513 | 0.954× | 127.44 / 133.65 / 93.33 |
| 1024 | 1536 | 3072 | 60.737 | 59.252 | 81.105 | 0.976× | 159.11 / 163.10 / 119.15 |
| 1024 | 2048 | 1536 | 49.302 | 47.156 | 65.383 | 0.956× | 130.67 / 136.62 / 98.53 |
| 1024 | 3072 | 1536 | 68.477 | 63.678 | 90.721 | 0.930× | 141.12 / 151.76 / 106.52 |
| 1024 | 4096 | 1536 | 98.204 | 94.242 | 121.883 | 0.960× | 131.21 / 136.72 / 105.72 |
| 1024 | 151936 | 1536 | 3496.096 | 3299.392 | 4304.128 | 0.944× | 136.71 / 144.86 / 111.04 |
| 1024 | 262144 | 1536 | 6136.672 | 5804.096 | 7411.904 | 0.946× | 134.38 / 142.08 / 111.26 |
| 2048 | 512 | 1536 | 29.270 | 28.240 | 33.277 | 0.965× | 110.05 / 114.07 / 96.80 |
| 2048 | 1024 | 1536 | 49.500 | 47.177 | 65.718 | 0.953× | 130.15 / 136.56 / 98.03 |
| 2048 | 1536 | 1024 | 50.429 | 48.097 | 64.496 | 0.954× | 127.75 / 133.95 / 99.89 |
| 2048 | 1536 | 3072 | 123.900 | 120.955 | 160.252 | 0.976× | 155.99 / 159.79 / 120.61 |
| 2048 | 2048 | 1536 | 100.542 | 95.836 | 122.595 | 0.953× | 128.15 / 134.45 / 105.10 |
| 2048 | 3072 | 1536 | 143.635 | 139.399 | 178.367 | 0.971× | 134.56 / 138.65 / 108.36 |
| 2048 | 4096 | 1536 | 198.284 | 196.102 | 237.611 | 0.989× | 129.96 / 131.41 / 108.45 |
| 2048 | 151936 | 1536 | 7580.672 | 6781.088 | 8700.992 | 0.895× | 126.10 / 140.97 / 109.86 |
| 2048 | 262144 | 1536 | 12807.104 | 11624.544 | 15010.720 | 0.908× | 128.78 / 141.88 / 109.87 |

Speedup = CuTe latency / CUTLASS latency; greater than 1 means CUTLASS is faster.

These measurements compare the implementations here, not a universal ranking of frameworks. CUTLASS uses a custom SM80-style collective and GemmUniversal composition compiled for SM121; both native backends use INT8 mma.sync and cp.async. The CUTLASS path uses the upstream vectorized epilogue. It does not invoke the direct CuTe kernel.

Correctness: independent CPU INT32 group dots with FP32 FMA rounding at up to 32×32 sampled outputs, full-output finite checks, and zero/restore CUDA graph replay for all selected kernels. Full per-round timings, candidates, hashes and build versions are in each case JSON.

## Native timing ranges

Ranges are the minimum and maximum of five round medians, not confidence intervals. Some large shapes vary by several percent between rounds; small performance gaps should not be generalized to other devices, builds or workloads.

| M | N | K | CUTLASS round range μs | CuTe round range μs | CUTLASS config | CuTe config |
|---:|---:|---:|---|---|---:|---:|
| 1024 | 512 | 1536 | 16.119–16.153 | 15.477–15.502 | 7 | 7 |
| 1024 | 1024 | 1536 | 29.178–29.283 | 28.169–28.570 | 3 | 3 |
| 1024 | 1536 | 1024 | 25.256–25.539 | 24.082–24.111 | 7 | 7 |
| 1024 | 1536 | 3072 | 58.343–60.989 | 58.500–59.308 | 7 | 7 |
| 1024 | 2048 | 1536 | 49.258–49.458 | 47.146–47.286 | 7 | 7 |
| 1024 | 3072 | 1536 | 68.381–69.327 | 63.636–66.277 | 6 | 7 |
| 1024 | 4096 | 1536 | 98.018–101.955 | 94.221–96.914 | 7 | 6 |
| 1024 | 151936 | 1536 | 3446.848–3549.632 | 3272.448–3402.784 | 7 | 15 |
| 1024 | 262144 | 1536 | 6043.584–6213.632 | 5650.880–5867.616 | 7 | 15 |
| 2048 | 512 | 1536 | 29.220–29.529 | 28.160–28.275 | 3 | 3 |
| 2048 | 1024 | 1536 | 49.338–49.672 | 47.146–47.537 | 7 | 7 |
| 2048 | 1536 | 1024 | 50.224–50.555 | 47.881–48.368 | 7 | 6 |
| 2048 | 1536 | 3072 | 120.188–124.541 | 120.812–121.679 | 7 | 7 |
| 2048 | 2048 | 1536 | 98.900–103.446 | 95.126–100.855 | 7 | 6 |
| 2048 | 3072 | 1536 | 143.417–143.914 | 133.021–146.034 | 7 | 7 |
| 2048 | 4096 | 1536 | 198.077–205.582 | 190.999–197.693 | 7 | 7 |
| 2048 | 151936 | 1536 | 7557.472–7793.824 | 6698.912–7223.488 | 7 | 15 |
| 2048 | 262144 | 1536 | 12773.280–13090.176 | 11582.368–11681.152 | 7 | 15 |