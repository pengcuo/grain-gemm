# Thor G256 INT8 batched GEMM: intermediate-width sweep

Measured 2026-10-21 19:32 CST. GPU frequency was reduced to **1.2 GHz (actual 1197 MHz)** and EMC fixed at **2750 MHz**. The benchmark verified minimum, maximum and current clock readings before and after timed samples.

Increasing intermediate width from 1024 to 4096 raises Down throughput from **118.30 to 190.99 TOPS** at 4096 rows per expert (**61.4%**). Gate/Up rises from **158.12 to 167.88 TOPS** (**6.2%**). Thus the small intermediate width substantially limits Down in this test, but increasing it alone does not take both projections above 200 TOPS.

All runs use **32 equal-length experts**, hidden width **H=1536**, G256 INT8 inputs/weights and BF16 output. Only intermediate width I changes. Gate/Up has `N=2I, K=1536`; Down has `N=1536, K=I`. Increasing I widens Gate/Up’s output but increases Down’s reduction dimension. Each replay launches one native batched GEMM across all experts, with no padding, routing, input packing, quantization, activation or combine.

## Results

`M` is rows **per expert**. Balanced Top-4 routing corresponds to `T=8M`: M512 maps to T4096, and M4096 to T32768. INT8 throughput is **TOPS**, using `2*32*M*N*K / time`. Cold measurements flush a 256 MB buffer outside the timed event. Warm measurements reuse the same inputs without a flush.

| M / expert | I | Gate/Up cold ms (TOPS) | Gate/Up warm ms (TOPS) | Down cold ms (TOPS) | Down warm ms (TOPS) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1024 | 1.2152 (84.83) | 1.1799 (87.36) | 0.7535 (68.40) | 0.7113 (72.46) |
| 512 | 2048 | 2.2166 (93.01) | 2.1703 (94.99) | 1.1570 (89.09) | 1.1081 (93.02) |
| 512 | 4096 | 4.2348 (97.36) | 4.1849 (98.52) | 1.9932 (103.43) | 1.9024 (108.37) |
| 4096 | 1024 | 5.2692 (156.50) | 5.2154 (158.12) | 3.5279 (116.87) | 3.4853 (118.30) |
| 4096 | 2048 | 10.6087 (155.46) | 10.5827 (155.85) | 5.0536 (163.18) | 5.0268 (164.05) |
| 4096 | 4096 | 19.5380 (168.83) | 19.6484 (167.88) | 8.7304 (188.91) | 8.6355 (190.99) |

I1024 values are the immediately preceding [equal-length benchmark](batched_gemm_equal_g256_emc2750.md), using the same script, native kernel and locked frequencies; I2048 and I4096 were measured in subsequent sequential runs. The widths were not interleaved in one timing session.

An I8192 run was stopped at the user’s request. Its partial results are excluded from the comparison; no further I8192 samples were collected after interruption.

## Validation and configurations

All 12 native configurations were screened over three rounds. The top two were additionally tested with swizzle 1/4/8 and raster 0/1/2. Each shape’s selected configuration was fixed for independent confirmation: three rounds of three samples per cache mode, reporting the median of round medians. Warm results were not separately autotuned.

All candidate configurations passed bit-exact BF16 comparisons against independent CPU INT32 G256 dot products with FP32 scaled accumulation, sampling 81 elements across three experts, three rows and nine columns. Input data spans the full signed INT8 range, with zero and negative scale cases. Winning outputs passed another sampled comparison after timing and a full-output finite-value check. This is sampled correctness validation, not a full numerical comparison.

Source, native wrapper, build metadata and binary fingerprints were unchanged throughout each completed run. The same benchmark and native fingerprint maps match across I1024, I2048 and I4096. No production kernel or dispatch changes were made for this sweep.

| I | M / expert | Projection | Config / swizzle / raster |
| ---: | ---: | --- | --- |
| 1024 | 512 | gate_up | 35 / 1 / 2 |
| 1024 | 512 | down | 36 / 8 / 1 |
| 1024 | 4096 | gate_up | 36 / 1 / 1 |
| 1024 | 4096 | down | 46 / 1 / 1 |
| 2048 | 512 | gate_up | 35 / 4 / 2 |
| 2048 | 512 | down | 36 / 4 / 2 |
| 2048 | 4096 | gate_up | 45 / 1 / 0 |
| 2048 | 4096 | down | 45 / 1 / 1 |
| 4096 | 512 | gate_up | 35 / 4 / 0 |
| 4096 | 512 | down | 36 / 1 / 2 |
| 4096 | 4096 | gate_up | 45 / 1 / 0 |
| 4096 | 4096 | down | 45 / 4 / 2 |

| I | Maximum observed GPU temperature |
| ---: | ---: |
| 2048 | 63.25 °C |
| 4096 | 65.0 °C |

The thermal guard pauses at 62 °C until 54 °C and aborts at 70 °C. Clock settings are read-only.

## Reproduction

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_batched_gemm.py \
  --experts 32 --rows 512 4096 --hidden 1536 --intermediate 2048 \
  --output /tmp/thor_batched_gemm_i2048.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_batched_gemm.py \
  --experts 32 --rows 512 4096 --hidden 1536 --intermediate 4096 \
  --output /tmp/thor_batched_gemm_i4096.json
```

Raw logs, JSON measurements and measured source/binary snapshots are archived outside Git in `/workspace/artifacts/thor_batched_gemm_intermediate_20261021/`.
