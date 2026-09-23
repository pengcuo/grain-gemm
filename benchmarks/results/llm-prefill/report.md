# LLM prefill GEMM benchmark: INT8 G256

> Historical measurement archive (2026-09-22). “Default” means the explicit backend selection recorded at measurement time, not today’s API defaults. All runs explicitly used BF16 output. The public API now defaults to BF16, and the three M=256 dispatch improvements are integrated. Raw JSON and CSV are preserved byte for byte.

GPU: **NVIDIA GB10**. Source: [results.json](results.json). 51 unique matrix shapes cover 54 workload uses.

Speedup means **Triton median time / CuTe median time**. Values above 1 favor CuTe; values below 1 favor Triton. Measurement-time forced-backend defaults and locally tuned configurations are compared separately; default CuTe timings do not describe automatic dispatch.

## Measurements by total token count

Backend columns show median latency in **µs**. Every row lists M, N, and K explicitly. T is the total dense prompt-token count; routed-expert M=T/8 assumes balanced routing. N/A means native CuTe does not support the shape.

The added **Equivalent TFLOPS (Triton / CuTe)** column uses `2 × M × N × K / (latency_us × 10^6)`, computed from unrounded timings. These are INT8 GEMMs: the physical unit is **TOPS**, and the equivalent-TFLOPS label is an operation-count convention with the same numerical value, not a measurement of floating-point arithmetic throughput.

Effective TOPS and detailed measurements are retained in [results.csv](results.csv) and the source JSON.

### Total prompt tokens T = 512

**Measurement-time forced-backend defaults**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 512 | 3,072 | 1,536 | 47.448 | 33.021 | 1.437× | 101.83 / 146.33 |
| K or V projection (`k_or_v`) | 512 | 512 | 1,536 | 9.809 | 15.062 | 0.651× | 82.10 / 53.47 |
| Fused QKV (`qkv`) | 512 | 4,096 | 1,536 | 66.478 | 47.955 | 1.386× | 96.91 / 134.34 |
| Attention output (`attention_out`) | 512 | 1,536 | 3,072 | 40.836 | 30.383 | 1.344× | 118.32 / 159.03 |
| Shared gate or up (`shared_gate_or_up`) | 512 | 1,024 | 1,536 | 19.926 | 15.606 | 1.277× | 80.83 / 103.21 |
| Shared fused gate + up (`shared_gate_up`) | 512 | 2,048 | 1,536 | 33.575 | 31.250 | 1.074× | 95.94 / 103.08 |
| Shared down (`shared_down`) | 512 | 1,536 | 1,024 | 16.940 | 12.848 | 1.318× | 95.08 / 125.35 |
| Router projection (`router`) | 512 | 32 | 1,536 | 5.136 | N/A | N/A | 9.80 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 64 | 1,024 | 1,536 | 9.579 | 7.883 | 1.215× | 21.02 / 25.54 |
| One expert fused gate + up (`expert_gate_up`) | 64 | 2,048 | 1,536 | 9.573 | 8.356 | 1.146× | 42.06 / 48.19 |
| One expert down (`expert_down`) | 64 | 1,536 | 1,024 | 7.182 | 5.943 | 1.208× | 28.03 / 33.88 |
| All-token LM head (151936) (`lm_head_151936`) | 512 | 151,936 | 1,536 | 2,264.204 | 1,914.212 | 1.183× | 105.54 / 124.84 |
| All-token LM head (262144) (`lm_head_262144`) | 512 | 262,144 | 1,536 | 3,905.001 | 3,326.208 | 1.174× | 105.59 / 123.96 |

**Offline locally tuned configurations**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 512 | 3,072 | 1,536 | 47.117 | 32.990 | 1.428× | 102.55 / 146.46 |
| K or V projection (`k_or_v`) | 512 | 512 | 1,536 | 9.429 | 8.804 | 1.071× | 85.41 / 91.47 |
| Fused QKV (`qkv`) | 512 | 4,096 | 1,536 | 64.928 | 47.936 | 1.354× | 99.22 / 134.40 |
| Attention output (`attention_out`) | 512 | 1,536 | 3,072 | 40.528 | 30.379 | 1.334× | 119.22 / 159.05 |
| Shared gate or up (`shared_gate_or_up`) | 512 | 1,024 | 1,536 | 19.418 | 15.612 | 1.244× | 82.94 / 103.16 |
| Shared fused gate + up (`shared_gate_up`) | 512 | 2,048 | 1,536 | 32.309 | 28.588 | 1.130× | 99.70 / 112.68 |
| Shared down (`shared_down`) | 512 | 1,536 | 1,024 | 16.607 | 12.846 | 1.293× | 96.98 / 125.38 |
| Router projection (`router`) | 512 | 32 | 1,536 | 3.827 | N/A | N/A | 13.15 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 64 | 1,024 | 1,536 | 5.867 | 5.636 | 1.041× | 34.32 / 35.72 |
| One expert fused gate + up (`expert_gate_up`) | 64 | 2,048 | 1,536 | 6.570 | 5.842 | 1.125× | 61.28 / 68.92 |
| One expert down (`expert_down`) | 64 | 1,536 | 1,024 | 4.538 | 4.362 | 1.040× | 44.37 / 46.16 |
| All-token LM head (151936) (`lm_head_151936`) | 512 | 151,936 | 1,536 | 2,228.210 | 1,877.020 | 1.187× | 107.25 / 127.32 |
| All-token LM head (262144) (`lm_head_262144`) | 512 | 262,144 | 1,536 | 3,816.350 | 3,276.359 | 1.165× | 108.04 / 125.85 |

### Total prompt tokens T = 1,024

**Measurement-time forced-backend defaults**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 1,024 | 3,072 | 1,536 | 91.394 | 65.649 | 1.392× | 105.74 / 147.20 |
| K or V projection (`k_or_v`) | 1,024 | 512 | 1,536 | 19.163 | 15.179 | 1.263× | 84.05 / 106.11 |
| Fused QKV (`qkv`) | 1,024 | 4,096 | 1,536 | 131.360 | 95.548 | 1.375× | 98.09 / 134.85 |
| Attention output (`attention_out`) | 1,024 | 1,536 | 3,072 | 81.977 | 60.568 | 1.353× | 117.88 / 159.55 |
| Shared gate or up (`shared_gate_or_up`) | 1,024 | 1,024 | 1,536 | 32.409 | 30.453 | 1.064× | 99.39 / 105.78 |
| Shared fused gate + up (`shared_gate_up`) | 1,024 | 2,048 | 1,536 | 69.406 | 51.880 | 1.338× | 92.82 / 124.18 |
| Shared down (`shared_down`) | 1,024 | 1,536 | 1,024 | 35.377 | 25.001 | 1.415× | 91.05 / 128.84 |
| Router projection (`router`) | 1,024 | 32 | 1,536 | 5.347 | N/A | N/A | 18.83 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 128 | 1,024 | 1,536 | 9.317 | 14.480 | 0.643× | 43.21 / 27.81 |
| One expert fused gate + up (`expert_gate_up`) | 128 | 2,048 | 1,536 | 9.714 | 15.405 | 0.631× | 82.90 / 52.28 |
| One expert down (`expert_down`) | 128 | 1,536 | 1,024 | 7.213 | 11.341 | 0.636× | 55.83 / 35.50 |
| All-token LM head (151936) (`lm_head_151936`) | 1,024 | 151,936 | 1,536 | 4,540.470 | 3,532.387 | 1.285× | 105.26 / 135.30 |
| All-token LM head (262144) (`lm_head_262144`) | 1,024 | 262,144 | 1,536 | 8,151.654 | 6,090.066 | 1.339× | 101.16 / 135.41 |

**Offline locally tuned configurations**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 1,024 | 3,072 | 1,536 | 88.666 | 66.097 | 1.341× | 108.99 / 146.20 |
| K or V projection (`k_or_v`) | 1,024 | 512 | 1,536 | 18.555 | 15.180 | 1.222× | 86.80 / 106.10 |
| Fused QKV (`qkv`) | 1,024 | 4,096 | 1,536 | 120.785 | 95.847 | 1.260× | 106.68 / 134.43 |
| Attention output (`attention_out`) | 1,024 | 1,536 | 3,072 | 80.703 | 60.465 | 1.335× | 119.74 / 159.82 |
| Shared gate or up (`shared_gate_or_up`) | 1,024 | 1,024 | 1,536 | 31.949 | 28.607 | 1.117× | 100.82 / 112.60 |
| Shared fused gate + up (`shared_gate_up`) | 1,024 | 2,048 | 1,536 | 64.263 | 48.302 | 1.330× | 100.25 / 133.38 |
| Shared down (`shared_down`) | 1,024 | 1,536 | 1,024 | 35.317 | 25.008 | 1.412× | 91.21 / 128.81 |
| Router projection (`router`) | 1,024 | 32 | 1,536 | 4.694 | N/A | N/A | 21.45 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 128 | 1,024 | 1,536 | 6.743 | 5.767 | 1.169× | 59.71 / 69.82 |
| One expert fused gate + up (`expert_gate_up`) | 128 | 2,048 | 1,536 | 9.398 | 8.814 | 1.066× | 85.69 / 91.37 |
| One expert down (`expert_down`) | 128 | 1,536 | 1,024 | 6.199 | 5.478 | 1.132× | 64.96 / 73.50 |
| All-token LM head (151936) (`lm_head_151936`) | 1,024 | 151,936 | 1,536 | 4,400.288 | 3,483.353 | 1.263× | 108.62 / 137.21 |
| All-token LM head (262144) (`lm_head_262144`) | 1,024 | 262,144 | 1,536 | 7,654.608 | 6,351.198 | 1.205× | 107.73 / 129.84 |

### Total prompt tokens T = 2,048

**Measurement-time forced-backend defaults**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 2,048 | 3,072 | 1,536 | 180.245 | 138.036 | 1.306× | 107.23 / 140.02 |
| K or V projection (`k_or_v`) | 2,048 | 512 | 1,536 | 32.270 | 30.417 | 1.061× | 99.82 / 105.90 |
| Fused QKV (`qkv`) | 2,048 | 4,096 | 1,536 | 244.221 | 196.283 | 1.244× | 105.52 / 131.29 |
| Attention output (`attention_out`) | 2,048 | 1,536 | 3,072 | 163.218 | 120.911 | 1.350× | 118.41 / 159.85 |
| Shared gate or up (`shared_gate_or_up`) | 2,048 | 1,024 | 1,536 | 65.411 | 48.384 | 1.352× | 98.49 / 133.15 |
| Shared fused gate + up (`shared_gate_up`) | 2,048 | 2,048 | 1,536 | 124.454 | 96.177 | 1.294× | 103.53 / 133.97 |
| Shared down (`shared_down`) | 2,048 | 1,536 | 1,024 | 68.562 | 49.616 | 1.382× | 93.96 / 129.85 |
| Router projection (`router`) | 2,048 | 32 | 1,536 | 7.072 | N/A | N/A | 28.47 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 256 | 1,024 | 1,536 | 9.651 | 14.500 | 0.666× | 83.44 / 55.54 |
| One expert fused gate + up (`expert_gate_up`) | 256 | 2,048 | 1,536 | 19.455 | 15.327 | 1.269× | 82.78 / 105.08 |
| One expert down (`expert_down`) | 256 | 1,536 | 1,024 | 8.023 | 10.936 | 0.734× | 100.38 / 73.64 |
| All-token LM head (151936) (`lm_head_151936`) | 2,048 | 151,936 | 1,536 | 8,883.984 | 7,186.200 | 1.236× | 107.60 / 133.02 |
| All-token LM head (262144) (`lm_head_262144`) | 2,048 | 262,144 | 1,536 | 15,610.208 | 12,530.720 | 1.246× | 105.65 / 131.62 |

**Offline locally tuned configurations**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 2,048 | 3,072 | 1,536 | 169.590 | 137.845 | 1.230× | 113.97 / 140.21 |
| K or V projection (`k_or_v`) | 2,048 | 512 | 1,536 | 31.554 | 28.486 | 1.108× | 102.09 / 113.08 |
| Fused QKV (`qkv`) | 2,048 | 4,096 | 1,536 | 237.806 | 196.036 | 1.213× | 108.36 / 131.45 |
| Attention output (`attention_out`) | 2,048 | 1,536 | 3,072 | 161.252 | 121.220 | 1.330× | 119.86 / 159.44 |
| Shared gate or up (`shared_gate_or_up`) | 2,048 | 1,024 | 1,536 | 64.206 | 48.363 | 1.328× | 100.34 / 133.21 |
| Shared fused gate + up (`shared_gate_up`) | 2,048 | 2,048 | 1,536 | 121.298 | 96.160 | 1.261× | 106.22 / 133.99 |
| Shared down (`shared_down`) | 2,048 | 1,536 | 1,024 | 62.076 | 49.307 | 1.259× | 103.78 / 130.66 |
| Router projection (`router`) | 2,048 | 32 | 1,536 | 6.815 | N/A | N/A | 29.54 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 256 | 1,024 | 1,536 | 9.299 | 8.746 | 1.063× | 86.60 / 92.08 |
| One expert fused gate + up (`expert_gate_up`) | 256 | 2,048 | 1,536 | 19.066 | 15.332 | 1.244× | 84.48 / 105.05 |
| One expert down (`expert_down`) | 256 | 1,536 | 1,024 | 8.017 | 8.018 | 1.000× | 100.45 / 100.43 |
| All-token LM head (151936) (`lm_head_151936`) | 2,048 | 151,936 | 1,536 | 8,674.573 | 6,739.321 | 1.287× | 110.20 / 141.84 |
| All-token LM head (262144) (`lm_head_262144`) | 2,048 | 262,144 | 1,536 | 15,041.509 | 11,650.284 | 1.291× | 109.65 / 141.56 |

### Total prompt tokens T = 4,096

**Measurement-time forced-backend defaults**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 4,096 | 3,072 | 1,536 | 361.788 | 282.847 | 1.279× | 106.84 / 136.66 |
| K or V projection (`k_or_v`) | 4,096 | 512 | 1,536 | 66.491 | 48.667 | 1.366× | 96.89 / 132.38 |
| Fused QKV (`qkv`) | 4,096 | 4,096 | 1,536 | 477.650 | 374.941 | 1.274× | 107.90 / 137.46 |
| Attention output (`attention_out`) | 4,096 | 1,536 | 3,072 | 326.095 | 248.824 | 1.311× | 118.54 / 155.35 |
| Shared gate or up (`shared_gate_or_up`) | 4,096 | 1,024 | 1,536 | 124.524 | 96.764 | 1.287× | 103.47 / 133.16 |
| Shared fused gate + up (`shared_gate_up`) | 4,096 | 2,048 | 1,536 | 245.400 | 190.791 | 1.286× | 105.01 / 135.07 |
| Shared down (`shared_down`) | 4,096 | 1,536 | 1,024 | 133.431 | 102.779 | 1.298× | 96.57 / 125.37 |
| Router projection (`router`) | 4,096 | 32 | 1,536 | 9.424 | N/A | N/A | 42.73 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 512 | 1,024 | 1,536 | 19.926 | 15.606 | 1.277× | 80.83 / 103.21 |
| One expert fused gate + up (`expert_gate_up`) | 512 | 2,048 | 1,536 | 33.575 | 31.250 | 1.074× | 95.94 / 103.08 |
| One expert down (`expert_down`) | 512 | 1,536 | 1,024 | 16.940 | 12.848 | 1.318× | 95.08 / 125.35 |
| All-token LM head (151936) (`lm_head_151936`) | 4,096 | 151,936 | 1,536 | 17,962.873 | 13,213.536 | 1.359× | 106.43 / 144.68 |
| All-token LM head (262144) (`lm_head_262144`) | 4,096 | 262,144 | 1,536 | 31,116.656 | 22,904.456 | 1.359× | 106.01 / 144.01 |

**Offline locally tuned configurations**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q projection (`q`) | 4,096 | 3,072 | 1,536 | 348.716 | 282.834 | 1.233× | 110.85 / 136.67 |
| K or V projection (`k_or_v`) | 4,096 | 512 | 1,536 | 66.001 | 48.639 | 1.357× | 97.61 / 132.45 |
| Fused QKV (`qkv`) | 4,096 | 4,096 | 1,536 | 470.685 | 374.571 | 1.257× | 109.50 / 137.60 |
| Attention output (`attention_out`) | 4,096 | 1,536 | 3,072 | 321.765 | 248.928 | 1.293× | 120.13 / 155.28 |
| Shared gate or up (`shared_gate_or_up`) | 4,096 | 1,024 | 1,536 | 122.529 | 96.721 | 1.267× | 105.16 / 133.22 |
| Shared fused gate + up (`shared_gate_up`) | 4,096 | 2,048 | 1,536 | 233.890 | 190.893 | 1.225× | 110.18 / 135.00 |
| Shared down (`shared_down`) | 4,096 | 1,536 | 1,024 | 121.786 | 102.662 | 1.186× | 105.80 / 125.51 |
| Router projection (`router`) | 4,096 | 32 | 1,536 | 9.265 | N/A | N/A | 43.46 / N/A |
| One expert gate or up (`expert_gate_or_up`) | 512 | 1,024 | 1,536 | 19.418 | 15.612 | 1.244× | 82.94 / 103.16 |
| One expert fused gate + up (`expert_gate_up`) | 512 | 2,048 | 1,536 | 32.309 | 28.588 | 1.130× | 99.70 / 112.68 |
| One expert down (`expert_down`) | 512 | 1,536 | 1,024 | 16.607 | 12.846 | 1.293× | 96.98 / 125.38 |
| All-token LM head (151936) (`lm_head_151936`) | 4,096 | 151,936 | 1,536 | 17,408.800 | 13,199.158 | 1.319× | 109.82 / 144.84 |
| All-token LM head (262144) (`lm_head_262144`) | 4,096 | 262,144 | 1,536 | 30,073.024 | 22,890.248 | 1.314× | 109.68 / 144.10 |

### Last-token LM heads (M = 1)

**Measurement-time forced-backend defaults**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last-token LM head (151936) (`lm_head_last_151936`) | 1 | 151,936 | 1,536 | 932.110 | N/A | N/A | 0.50 / N/A |
| Last-token LM head (262144) (`lm_head_last_262144`) | 1 | 262,144 | 1,536 | 1,608.564 | N/A | N/A | 0.50 / N/A |

**Offline locally tuned configurations**

| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last-token LM head (151936) (`lm_head_last_151936`) | 1 | 151,936 | 1,536 | 899.531 | N/A | N/A | 0.52 / N/A |
| Last-token LM head (262144) (`lm_head_last_262144`) | 1 | 262,144 | 1,536 | 1,548.044 | N/A | N/A | 0.52 / N/A |

## Timing and validation

Sweep resumed after host restart; completed cases preserved, remaining shapes measured with unchanged kernel binaries/software/timing settings. The sweep therefore spans multiple sessions rather than one uninterrupted run.

The tables use medians of the recorded round medians. Each backend's recorded minimum and maximum, round count, effective TOPS, and both speedup ratios were checked. Workload coverage, projection dimensions, per-expert occupancy, and native-CUDA eligibility were also checked before rendering. Effective TOPS is `2 × M × N × K / (median_ms × 10^9)`.

Timing metadata from the benchmark:

```json
{
  "rounds": 5,
  "rep_ms": 50,
  "tune_rounds": 3,
  "tune_rep_ms": 10,
  "warmup_calls": 20,
  "method": "triton.testing.do_bench_cudagraph",
  "statistic": "median of per-round medians",
  "order": "rotating per round",
  "cache_policy": "repeated inputs; no cache flush",
  "scope": "Preallocated-output GPU GEMM only; includes INT32 group reduction, FP32 scaling/accumulation, BF16 output; excludes input quantization, compilation, tuning, host dispatch and allocation",
  "tuning": "Select each backend separately from existing candidates; final measurement uses fresh data and a separate timing pass"
}
```

All per-round measurements, sampled correctness metadata, tuning candidates, and selected configuration dictionaries remain available in the source JSON.

Default and tuned selections are timed separately, including when they select the same configuration. Small differences between those repeated measurements do not demonstrate a tuning improvement; inspect the selected configurations and round variability before attributing a change to tuning. Resource-limited non-default tuning candidates may be skipped; the candidate search is local and does not establish a globally optimal configuration.

## Workload interpretation

- The model uses hidden size 1536, 24 layers, 24 query heads and 4 KV heads with head dimension 128. Query width is therefore 3072; each key/value width is 512.
- FFNs assume gated SiLU (SwiGLU): two 1536→1024 gate/up projections and one 1024→1536 down projection. There are 32 routed experts, top-4 routing, and one shared expert.
- Dense/shared projections use M=T. A routed expert uses M=T×4/32=T/8 under balanced routing. Its row measures one ordinary GEMM, not grouped-MoE performance or end-to-end expert dispatch and combination. Real occupancy imbalance is not represented.
- Q versus fused QKV, and separate gate/up versus fused gate+up, are implementation alternatives. Their rows must not be added together. The K-or-V and gate-or-up rows represent one projection each.
- Attention QK/PV matrix multiplications, softmax, activations, routing, and communication are outside this linear-projection benchmark.
- All-token LM heads (M=T) represent optional prompt-logprob or full-logit workloads. Typical single-sequence generation after prefill needs only the last-token head (M=1), which is unsupported by this native CuTe kernel and is shown separately.
- Inputs are synthetic INT8 tensors with G256 quantization groups, FP32 scales, and BF16 outputs. Quantization cost is excluded. These measurements establish neither model accuracy nor end-to-end inference latency.
- Native CuTe requires SM121, aligned contiguous A and B.T, contiguous scales, M/N multiples of 64, and K a multiple of 256. N=32 router and M=1 last-token head cases are N/A; Triton still runs them.
- At measurement time, automatic dispatch selected Triton for these rectangular LLM shapes. CuTe measurements used the explicit native backend. The three measured M=256 shapes have since been integrated into automatic dispatch; see [the M256 investigation](../m256-dispatch/README.md). The timings here retain the original configuration selections.

## Shape reuse

Identical [M,N,K] shapes are timed once and reused for every matching workload. This includes shared and routed-expert projections where their token counts happen to produce the same GEMM shape.

| [M, N, K] | Workload uses |
| :--- | :--- |
| [512, 1,024, 1,536] | `shared_gate_or_up` T=512; `expert_gate_or_up` T=4,096 |
| [512, 1,536, 1,024] | `shared_down` T=512; `expert_down` T=4,096 |
| [512, 2,048, 1,536] | `shared_gate_up` T=512; `expert_gate_up` T=4,096 |
