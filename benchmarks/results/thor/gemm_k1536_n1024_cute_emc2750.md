# Thor K1536/N1024 single GEMM: CuTe follow-up

The preceding [rectangular GEMM measurement](gemm_k1536_n1024_emc2750.md) omitted the independent CUDA/CuTe backend from its candidate set. Its native results must not be interpreted as the best result across every backend. This follow-up includes all **30 CuTe configurations**, alongside the previous 47 candidates, using the same shapes and locked clocks.

GPU frequency is reduced to **1.2 GHz (actual 1197 MHz)**; EMC is **2750 MHz**. Each operation is one G256 INT8 GEMM `A[M,1536] @ B[1536,1024]`, with FP32 scales and BF16 output. No quantization, packing, routing, activation or combine is timed.

## Same-run repeated-input confirmation

| M | CuTe warm selection | CuTe µs | CuTe TOPS | Overall warm-selected candidate | Its µs (TOPS) | Auto TOPS |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 1024 | cute_27 | 35.968 | 89.56 | expert_45 | 33.408 (96.42) | 102.93 |
| 2048 | cute_27 | 57.920 | 111.23 | expert_36 | 49.056 (131.33) | 35.25 |
| 4096 | cute_27 | 92.448 | 139.37 | expert_36 | 82.816 (155.58) | 36.28 |
| 8192 | cute_27 | 168.640 | 152.81 | expert_36 | 155.808 (165.39) | 36.20 |

Candidate selection precedes independent confirmation. Screening and confirmation can rank close candidates differently; the table retains the screened choices rather than choosing new winners after confirmation. `expert_*` denotes the native expert entry point with **E=1** and all rows valid, so it executes one mathematical GEMM without routing or replication.

The original CuTe path remains available and passes validation. It does not reach 200 TOPS in these measured shapes at the locked clocks. Auto still uses CUTLASS #32 at M1024 and falls back to portable Triton at M2048/4096/8192; no production dispatch is changed by this experiment.

## Cold-cache confirmation

A 256 MB flush runs outside each timed replay. CuTe is selected independently for this mode.

| M | CuTe cold selection | CuTe µs (TOPS) |
| ---: | --- | ---: |
| 1024 | cute_27 | 84.832 (37.97) |
| 2048 | cute_27 | 108.800 (59.21) |
| 4096 | cute_27 | 159.616 (80.72) |
| 8192 | cute_26 | 261.920 (98.39) |

## Method and validation

All 77 candidates are screened for three rounds in each cache mode: public Auto, all 30 independent CuTe configurations, all 34 single-GEMM CUTLASS configurations and all 12 E=1 expert configurations (swizzle 1, raster 0). Auto, the overall cold/warm selections and the CuTe cold/warm selections are then independently confirmed with three rounds of three samples per cache mode. Latency is the median of round medians. Each graph replay contains one GEMM, with no long sustained warmup. Throughput is `2*M*N*K / time` in INT8 TOPS.

All **308 candidate/shape combinations** passed sampled bit-exact BF16 comparisons against independent CPU INT32 per-G256 dot products with FP32 scaled accumulation, including zero and negative scales. Each check samples 27 elements across three rows and nine columns, before and after graph capture. The overall cold-selected output is rechecked after timing and its entire output is finite. These are sampled numerical comparisons, not exhaustive output comparisons.

All six clock readings match the locks before and after samples. Maximum observed temperature: 55.718 °C. The guard pauses at 62 °C until 54 °C and aborts at 70 °C. Benchmark/native source, configuration and binary fingerprints were stable throughout the run.

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_rectangular.py \
  --rows 1024 2048 4096 8192 --n 1024 --k 1536 \
  --output /tmp/thor_gemm_k1536_n1024_with_cute.json
```

Raw JSON, logs and measured source/native snapshots are stored outside Git at `/workspace/artifacts/thor_gemm_k1536_n1024_cute_20261021/`.
