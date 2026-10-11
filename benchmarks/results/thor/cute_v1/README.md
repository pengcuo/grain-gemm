# Thor SM110 validation and performance

Validated on NVIDIA Thor, compute capability 11.0, 20 SMs. PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102, Triton 3.6.0, CUDA runtime 13.2. Native CUDA builds use CUDA 13.2 and `sm_110a`; driver 620.54. Clocks and power policy were not changed.

The public `int8_gemm` / `get_kernel_config` interfaces, scale ordering and numerical contract are the same as GB10. Activations are INT8 row-major `[M,K]`; logical weights `[K,N]` are a transposed contiguous `[N,K]` tensor. Scales are contiguous FP32 `scale_a[M,K/256]` and `scale_b[K/256,N]`. No input or scale packing, quantization or copies occur in the GEMM API. INT32 sums are completed for each G256, then FP32 FMA accumulates scaled groups; conversion to BF16 occurs once.

The Thor CUDA implementation uses TMA, INT8 `tcgen05` MMA, two Tensor Memory buffers, a shared FP32 scale ring, and coalesced 128-bit BF16 stores. The independently built CUTLASS collective retains its warp MMA implementation on SM110. GB10 CUDA source and tuning tables are unchanged.

## Correctness

- Full suite: **799 passed, 13 skipped**. Skips cover GB10-specific hardware checks and measurements.
- Independent CPU INT32 group-dot references cover G32/G64/G128/G256, M/N/K tails, positive-stride views and BF16/FP16/FP32 outputs.
- All 26 Thor CUDA configurations and all 32 CUTLASS configurations are checked. Thor TMEM tests include minimum K, pipeline wrap, partial tile stripes, signed/zero/distinct scales, INT8 extrema and exact `2^-36` cancellation.
- CUDA Graph replay, updates to operands/scales, non-default streams, M64 tiles, 16-byte operand offsets and 4-byte scale offsets are validated.
- CUDA/CuTe and CUTLASS: memcheck reports **0 errors**; racecheck reports **0 errors / 0 warnings**. Logs are included here.

Memcheck uses `--target-processes application-only --report-api-errors explicit`. The default extended-API mode reports an optional TileIR compiler initialization failure in this container at `cudaGetDeviceCount`; the native kernels use NVCC cubins and do not use TileIR. The extended diagnostic log is retained as `memcheck_extended_api.log`; explicit CUDA API and kernel memory errors remain checked. Tracking only the application also avoids instrumenting Triton compiler subprocesses.

## Independent performance comparison

An offline search measures 8 Triton, up to 26 CUDA and 4 CUTLASS candidates over 3 rounds (20 ms captured batches). The table below is a **fresh 5-round comparison**, using 20 ms captured batches and rotating implementation order. The installed Triton helper replays each batch ten times; the median of round medians is reported separately for each shape. Each candidate is checked against sampled independent CPU INT32 group dots before timing. Auto/CUDA graph replay is additionally checked in the comparison.

Timing covers prequantized GEMM, group scaling and BF16 output conversion. It excludes input quantization, compilation, tuning, warmup and host dispatch. Repeated inputs are used without cache flushes. These measurements describe this machine and these shapes; they do not establish model accuracy, end-to-end inference speedups, or a universal ranking for unmeasured shapes.

| M × N × K | Auto | Auto µs | CUDA µs | Tuned Triton µs | BF16 µs | Auto / original Triton |
|---|---|---:|---:|---:|---:|---:|
| 1024 × 1024 × 1024 | triton | 35.44 | 37.10 | 35.59 | 18.04 | 1.38× |
| 2048 × 2048 × 2048 | cuda #19 | 198.51 | 198.37 | 224.82 | 113.32 | 1.79× |
| 3072 × 3072 × 3072 | cuda #19 | 775.73 | 776.24 | 735.17 | 501.65 | 1.48× |
| 4096 × 4096 × 4096 | cuda #20 | 1659.85 | 1657.50 | 1748.53 | 1099.99 | 1.58× |
| 6144 × 6144 × 6144 | cuda #20 | 5254.75 | 5263.66 | 5854.01 | 3890.35 | 1.62× |
| 8192 × 8192 × 8192 | cuda #20 | 11981.17 | 11983.65 | 13911.30 | 9798.30 | 1.78× |
| 12288 × 12288 × 12288 | cuda #25 | 39968.82 | 39963.25 | 46182.05 | 32774.03 | 1.71× |
| 16384 × 16384 × 16384 | cuda #25 | 94516.43 | 94495.06 | 108811.46 | 82761.92 | 1.78× |
| 256 × 1024 × 1536 | cuda #19 | 13.58 | 13.58 | 14.99 | 9.81 | 1.50× |
| 256 × 1536 × 1024 | triton | 14.80 | 16.78 | 14.81 | 8.80 | 1.38× |
| 256 × 2048 × 1536 | cuda #19 | 26.00 | 26.00 | 27.73 | 18.12 | 1.44× |
| 1024 × 1024 × 1536 | triton | 49.87 | 47.76 | 49.87 | 27.09 | 1.40× |
| 1024 × 1536 × 1024 | triton | 49.20 | 49.20 | 49.18 | 32.36 | 1.47× |
| 1024 × 4096 × 1536 | cuda #19 | 161.43 | 161.45 | 176.04 | 106.34 | 1.57× |
| 2048 × 2048 × 1536 | cuda #19 | 161.17 | 161.18 | 176.39 | 110.51 | 1.67× |
| 2048 × 151936 × 1536 | triton | 13347.33 | 13963.82 | 13347.17 | 8221.74 | 1.44× |

`original Triton` is the portable 32×64, 4-warp, 2-stage path used on Thor before this change. `tuned Triton` uses the best measured candidate for that exact shape. The BF16 column uses `torch.mm` on the original BF16 inputs and computes a different, unquantized operation. The full CSV also includes CUTLASS latency; JSON retains every round, configuration and correctness check.

The 3072 square showed run-to-run variability: the five-round comparison above recorded CUDA 776 µs versus Triton 735 µs, while a separate seven-round, 30 ms recheck recorded CUDA 655 µs versus Triton 728 µs (auto 655 µs). Both runs retain their raw measurements; the tuning-time CUDA selection is retained, but it is not a guarantee that CUDA wins every run. See [3072 recheck](recheck_3072.json). Reproduce it with `python benchmarks/bench_thor.py --shapes 3072x3072x3072 --rounds 7 --rep-ms 30 --output benchmarks/results/thor/recheck_3072.json`.

Automatic CUDA selection requires a measured advantage of at least 5% over tuned Triton. Unmeasured shapes retain portable Triton; explicit CUDA uses compatible native defaults. Unavailable/stale native libraries, other group sizes, other output types and unsupported layouts fall back to Triton in auto mode. Native layout restrictions match GB10: G256, BF16, M/N multiples of 64, positive K multiple of 256, 16-byte-aligned operands and contiguous FP32 scales.

## Reproduce

From the repository root, with the pinned CUTLASS checkout `098de2a652cf8f00fd70b2df54051c7eccbb855a`:

```bash
python -m pip install -e ".[runtime,test]"
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_110
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
python -m pytest -q
compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python tests/sanitize_kernels.py
compute-sanitizer --tool racecheck --target-processes application-only --error-exitcode 1 python tests/sanitize_kernels.py
compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python tests/sanitize_cutlass.py
compute-sanitizer --tool racecheck --target-processes application-only --error-exitcode 1 python tests/sanitize_cutlass.py
python benchmarks/bench_thor.py --tune
python benchmarks/bench_thor.py --rounds 5 --rep-ms 20
```

Builds are local and excluded from distributions; each backend installs one GPU target at a time. Restart Python after rebuilding a loaded library. The wheel contains SM110 source, dispatch data and helpers, and excludes native binaries.

[Raw tuning](tuning.json) · [Fresh comparison](comparison.json) · [Per-shape CSV](comparison.csv) · [Full pytest log](pytest.log)
