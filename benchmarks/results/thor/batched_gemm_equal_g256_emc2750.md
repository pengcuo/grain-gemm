# Equal-length expert batched G256 INT8 GEMM on Thor

Measured 2026-10-21 19:28 CST. GPU frequency was reduced to **1.2 GHz (actual 1197 MHz)**; EMC was fixed at **2750 MHz**. All minimum, maximum and current clock readings matched these settings before and after timed samples.

The current native implementation does **not reach 200 TOPS** on these MoE projection shapes. With 32 equal-length experts, the best repeated-input result is **169.89 TOPS for Gate/Up** and **123.88 TOPS for Down**. These results concern the tested shapes, implementation and clocks; they do not establish a general INT8 throughput ceiling.

## Scope and layouts

Each timed graph executes one native batched GEMM covering all experts. All rows are valid (`counts=None`); there is no routing, padding, quantization, activation, output combine, or Python loop over experts inside the timed interval. The benchmark reuses the existing native kernel rather than changing production dispatch.

- INT8 A: `[E,M,K]`, contiguous.
- INT8 B: mathematical `[E,K,N]`, a transpose view of contiguous Linear weights `[E,N,K]`.
- FP32 scales: A `[E,M,K/256]`, B `[E,K/256,N]`, both contiguous.
- BF16 output: `[E,M,N]`, contiguous.
- Gate/Up: `K=1536, N=2048`; Down: `K=1024, N=1536`.

`M` below is the number of rows **per expert**, not the input-token count. For balanced Top-4 routing with 32 experts, the corresponding input-token count is `8M`. Throughput is INT8 **TOPS**, calculated as `2*E*M*N*K / seconds / 1e12`, without a sparsity multiplier.

## 32-expert results

Cold measurements clear a 256 MB buffer before each timed replay; the clearing itself is excluded. Warm measurements reuse the same tensors with no cache flush. Both columns use the same configuration selected in a separate cold screening phase.

| Rows / expert | Equivalent Top-4 tokens | Gate/Up cold ms (TOPS) | Gate/Up warm ms (TOPS) | Down cold ms (TOPS) | Down warm ms (TOPS) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4096 | 1.2152 (84.83) | 1.1799 (87.36) | 0.7535 (68.40) | 0.7113 (72.46) |
| 1024 | 8192 | 1.7828 (115.64) | 1.7372 (118.68) | 1.1611 (88.78) | 1.1124 (92.66) |
| 2048 | 16384 | 2.9768 (138.51) | 2.9338 (140.54) | 1.9448 (106.00) | 1.9034 (108.31) |
| 4096 | 32768 | 5.2692 (156.50) | 5.2154 (158.12) | 3.5279 (116.87) | 3.4853 (118.30) |
| 8192 | 65536 | 9.7660 (168.88) | 9.7080 (169.89) | 6.7017 (123.05) | 6.6566 (123.88) |

## Single-expert controls

These controls keep the same N and K and use the same native implementation. `E=1, M=4096` matches each expert’s projection shape. `E=1, M=131072` matches the total row count of `E=32, M=4096`, but reuses one weight matrix across all rows; it is a different weight-reuse workload. They are not measurements of the public single-GEMM dispatcher or a large square GEMM.

| E | M | Projection | Cold ms (TOPS) | Warm ms (TOPS) |
| ---: | ---: | --- | ---: | ---: |
| 1 | 4096 | gate_up | 0.2170 (118.74) | 0.1507 (170.98) |
| 1 | 4096 | down | 0.1653 (77.94) | 0.1065 (120.95) |
| 1 | 131072 | gate_up | 4.5744 (180.27) | 4.5194 (182.47) |
| 1 | 131072 | down | 3.2357 (127.43) | 3.1841 (129.49) |

## Interpretation

Equal expert lengths remove routing imbalance and padded work, yet the tested projections remain below 200 TOPS. The control results show that removing the expert batch dimension alone does not make these N/K shapes reach 200 TOPS in this implementation. A single-GEMM result from another shape or dispatcher is not a matched control.

BF16 intermediate writes are substantial. At `E=32, M=8192`, Gate/Up writes 1,073,741,824 output bytes and Down writes 805,306,368 output bytes. Counting each A, B and scale tensor once plus one output write gives logical tensor bytes divided by cold latency of approximately 162.3 GB/s and 168.5 GB/s, respectively. These are **not measured DRAM bandwidth or traffic**: tile reuse and caches change physical transactions. The values suggest bandwidth pressure worth profiling; they do not prove bandwidth saturation. A fused activation/output-quantization path changes this IO workload and must be measured separately.

## Validation and timing

All 12 native tile configurations were screened over three rounds. The two leading configurations were additionally screened with swizzle 1/4/8 and raster 0/1/2. The selected configuration was held fixed for independent cold and warm confirmation: three rounds of three samples, reporting the median of round medians. Each replay contains one GEMM; allocation, input preparation, compilation and graph capture are excluded. Warm results were not separately autotuned.

Every tested configuration passed sampled, bit-exact BF16 checks against independent CPU INT32 G256 dots with FP32 scaled accumulation before timing. Inputs span the full signed INT8 range; scale tests include zeros and negative values. Each 32-expert case checks 81 output elements across three experts, three rows and nine columns; each single-expert case checks 27. The winning output is checked again after timing, and every output element is checked for finiteness. This is sampled numerical validation, not an exhaustive output comparison.

Benchmark, native source, wrapper, build metadata and binary SHA-256 hashes were stable throughout each run. Maximum observed temperatures were 63.75 °C (32 experts) and 62.687 °C (single expert). The guard pauses at 62 °C until 54 °C and aborts at 70 °C; no clocks are changed.

| E | M | Projection | Configuration / swizzle / raster |
| ---: | ---: | --- | --- |
| 32 | 512 | gate_up | 35 / 1 / 2 |
| 32 | 512 | down | 36 / 8 / 1 |
| 32 | 1024 | gate_up | 36 / 4 / 1 |
| 32 | 1024 | down | 34 / 1 / 0 |
| 32 | 2048 | gate_up | 36 / 1 / 1 |
| 32 | 2048 | down | 46 / 1 / 0 |
| 32 | 4096 | gate_up | 36 / 1 / 1 |
| 32 | 4096 | down | 46 / 1 / 1 |
| 32 | 8192 | gate_up | 36 / 1 / 0 |
| 32 | 8192 | down | 46 / 1 / 1 |
| 1 | 4096 | gate_up | 36 / 1 / 0 |
| 1 | 4096 | down | 46 / 1 / 0 |
| 1 | 131072 | gate_up | 36 / 1 / 0 |
| 1 | 131072 | down | 36 / 4 / 0 |

## Reproduction

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_batched_gemm.py \
  --experts 32 --rows 512 1024 2048 4096 8192 \
  --output /tmp/thor_batched_gemm_equal_g256.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_batched_gemm.py \
  --experts 1 --rows 4096 131072 \
  --output /tmp/thor_batched_gemm_single_control.json
```

Raw measurements, logs and measured source/binary snapshots are stored outside Git in `/workspace/artifacts/thor_batched_gemm_equal_20261021/`. A valid native batched GEMM build is required.
