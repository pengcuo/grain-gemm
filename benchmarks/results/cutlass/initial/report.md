# GB10 G256: CUTLASS collective vs direct CuTe vs Triton

Pure prequantized INT8 GEMM, FP32 group scales and accumulation, BF16 output. Each backend is tuned separately on the same input; selected candidates are timed again on fresh input. CUDA graphs use warm caches and preallocated output. Compilation, tuning, quantization, allocation and host dispatch are excluded.

Latency is μs. Equivalent TFLOPS uses 2MNK/time; the actual INT8 throughput unit is TOPS.

| M | N | K | CUTLASS μs | CuTe μs | Triton μs | CUTLASS / CuTe speedup | Equivalent TFLOPS: CUTLASS / CuTe / Triton |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 1024 | 1536 | 9.076 | 8.716 | 9.376 | 0.960× | 88.73 / 92.39 / 85.89 |
| 256 | 1536 | 1024 | 8.209 | 7.808 | 7.897 | 0.951× | 98.10 / 103.13 / 101.98 |
| 256 | 2048 | 1536 | 16.061 | 15.420 | 19.348 | 0.960× | 100.28 / 104.45 / 83.24 |
| 1024 | 1024 | 1024 | 22.347 | 20.598 | 24.676 | 0.922× | 96.10 / 104.26 / 87.03 |
| 2048 | 2048 | 2048 | 123.328 | 118.014 | 157.137 | 0.957× | 139.30 / 145.57 / 109.33 |
| 3072 | 3072 | 3072 | 368.451 | 359.787 | 467.053 | 0.976× | 157.37 / 161.16 / 124.14 |
| 4096 | 4096 | 4096 | 855.840 | 833.856 | 1087.248 | 0.974× | 160.59 / 164.82 / 126.41 |
| 5120 | 5120 | 5120 | 1675.093 | 1626.229 | 2010.160 | 0.971× | 160.25 / 165.07 / 133.54 |
| 6144 | 6144 | 6144 | 2813.024 | 2737.216 | 3391.616 | 0.973× | 164.90 / 169.46 / 136.77 |
| 7168 | 7168 | 7168 | 4426.688 | 4281.184 | 5384.640 | 0.967× | 166.40 / 172.05 / 136.79 |
| 8192 | 8192 | 8192 | 6594.400 | 6447.296 | 8023.072 | 0.978× | 166.73 / 170.54 / 137.04 |
| 9216 | 9216 | 9216 | 9288.544 | 9009.984 | 11285.568 | 0.970× | 168.54 / 173.75 / 138.72 |
| 10240 | 10240 | 10240 | 12858.528 | 12346.976 | 15588.384 | 0.960× | 167.01 / 173.93 / 137.76 |
| 11264 | 11264 | 11264 | 17194.016 | 16504.736 | 21478.559 | 0.960× | 166.24 / 173.18 / 133.08 |
| 12288 | 12288 | 12288 | 22157.375 | 21287.104 | 27803.808 | 0.961× | 167.48 / 174.32 / 133.47 |
| 13312 | 13312 | 13312 | 28556.864 | 27400.064 | 36085.953 | 0.959× | 165.21 / 172.19 / 130.74 |
| 14336 | 14336 | 14336 | 35692.513 | 34230.305 | 42911.263 | 0.959× | 165.10 / 172.15 / 137.32 |
| 15360 | 15360 | 15360 | 43661.407 | 41914.497 | 54225.086 | 0.960× | 166.00 / 172.92 / 133.66 |
| 16384 | 16384 | 16384 | 54442.047 | 51976.257 | 64676.544 | 0.955× | 161.57 / 169.23 / 136.00 |

Speedup = CuTe latency / CUTLASS latency; greater than 1 means CUTLASS is faster.

These measurements compare the implementations here, not a universal ranking of frameworks. CUTLASS uses a custom SM80-style collective and GemmUniversal composition compiled for SM121; both native backends use INT8 mma.sync and cp.async. The CUTLASS path uses the upstream vectorized epilogue. It does not invoke the direct CuTe kernel.

Correctness: independent CPU INT32 group dots with FP32 FMA rounding at up to 32×32 sampled outputs, full-output finite checks, and zero/restore CUDA graph replay for all selected kernels. Full per-round timings, candidates, hashes and build versions are in each case JSON.

## Native timing ranges

Ranges are the minimum and maximum of five round medians, not confidence intervals. Some large shapes vary by several percent between rounds; small performance gaps should not be generalized to other devices, builds or workloads.

| M | N | K | CUTLASS round range μs | CuTe round range μs | CUTLASS config | CuTe config |
|---:|---:|---:|---|---|---:|---:|
| 256 | 1024 | 1536 | 9.074–9.081 | 8.715–8.727 | 3 | 3 |
| 256 | 1536 | 1024 | 8.204–8.236 | 7.806–7.813 | 5 | 5 |
| 256 | 2048 | 1536 | 16.041–16.066 | 15.412–15.473 | 7 | 7 |
| 1024 | 1024 | 1024 | 22.314–22.454 | 20.523–20.641 | 4 | 2 |
| 2048 | 2048 | 2048 | 123.223–123.554 | 117.988–118.672 | 6 | 6 |
| 3072 | 3072 | 3072 | 350.691–370.602 | 338.802–360.071 | 7 | 7 |
| 4096 | 4096 | 4096 | 802.285–880.762 | 793.536–871.216 | 7 | 7 |
| 5120 | 5120 | 5120 | 1630.475–1683.275 | 1534.293–1629.195 | 7 | 15 |
| 6144 | 6144 | 6144 | 2625.696–2819.136 | 2540.448–2742.336 | 7 | 15 |
| 7168 | 7168 | 7168 | 4096.832–4439.104 | 3924.000–4295.200 | 7 | 15 |
| 8192 | 8192 | 8192 | 6169.216–6643.072 | 5936.512–6478.016 | 7 | 7 |
| 9216 | 9216 | 9216 | 9273.024–9304.128 | 8981.536–9144.608 | 7 | 15 |
| 10240 | 10240 | 10240 | 12569.888–12866.816 | 12309.600–12544.928 | 7 | 15 |
| 11264 | 11264 | 11264 | 17170.624–17357.471 | 16474.752–16584.192 | 7 | 15 |
| 12288 | 12288 | 12288 | 22135.937–22174.175 | 21155.071–21530.016 | 7 | 15 |
| 13312 | 13312 | 13312 | 28517.441–29049.280 | 27348.991–27588.608 | 7 | 15 |
| 14336 | 14336 | 14336 | 35680.286–35987.617 | 34094.498–34337.856 | 7 | 15 |
| 15360 | 15360 | 15360 | 43628.159–43945.057 | 41769.375–42143.806 | 7 | 15 |
| 16384 | 16384 | 16384 | 54370.399–54734.848 | 51951.744–52871.872 | 7 | 16 |