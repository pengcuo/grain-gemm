> Historical results before the independent SM110 CuTe warp-specialized pipeline. See the parent directory for the current implementation.

# Thor SM110 validation and performance

Validated on NVIDIA Thor, capability 11.0, 20 SMs. PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102, Triton 3.6.0, CUDA runtime 13.2; NVCC 13.2, target `sm_110a`, driver 620.54. Clocks and power policy were not changed.

The final public `backend="auto"` API is faster than PyTorch BF16 on **17/19 measured shapes**. Per-shape BF16/auto ratios range from **0.91× to 2.31×**; peak effective INT8 throughput is **223.84 TOPS**. These are individual shape results, not an average.

## Implementation and compatibility

CUTLASS configs 32/33 adapt the Thor G256 pipeline from `nvidia-cosmos/cosmos-int8-gemm`, commit `8977529aa23f70f02bfc36559599f00d59792996`. A two-SM 256×256×256 tile combines TMA INT8 loads, UMMA/TMEM, three fragment buffers, eight promotion warps, vector shared scale loads and 48/216/216 register redistribution. Config 32 retains the structured `EpilogueTileAuto`; config 33 is an explicit 128×16 candidate. The initial integration omitted the Cosmos release flag `-DNDEBUG`, which caused extra spills; the production builder now enforces it. The [incomplete diagnostic](../diagnostics/README.md) is retained separately.

The public `int8_gemm` and `get_kernel_config` APIs match GB10. A is INT8 row-major `[M,K]`, and logical B `[K,N]` is a transposed contiguous `[N,K]` tensor. Scales remain contiguous FP32 `scale_a[M,K/256]` and `scale_b[K/256,N]`. The Cosmos activation-scale layout is specialized to K-major; the API performs no transpose, packing, quantization or hidden copy. Each complete G256 INT32 dot contributes through `fmaf(float(partial), scale_a * scale_b, total)` to the FP32 sum, then BF16 conversion occurs once. Experimental biased and separable-scale modes are disabled.

Kernel parameters and descriptors are rebuilt for each invocation. The current CLC scheduler requires zero scratch bytes; the ABI supports caller-owned scratch through PyTorch if a configuration requires it. CUDA Graph and concurrent streams share no cached native handle or pointer-dependent state. Four-byte scale offsets remain valid: the vector weight-scale path uses the measured CUDA/Triton fallback in auto mode or the legacy scalar-compatible CUTLASS path when explicitly requested.

GB10 sources, both GB10 tuning tables, all existing configuration IDs and the legacy C ABI remain unchanged. The [earlier Thor CuTe measurements](../cute_v1/README.md) are preserved as historical results.

## Correctness

- Full suite after final dispatch installation: **839 passed, 13 skipped**.
- Independent CPU INT32 group-dot references cover all 26 Thor CUDA and 34 CUTLASS configurations. New pipeline tests include K256/512/768/4352, minimum/partial tiles, signed/zero/distinct scales, INT8 extrema and exact `2^-36` cancellation.
- Operand/scale offsets, CUDA Graph replay with updated operands and scales, non-default streams, concurrent streams and missing-library fallbacks pass.
- Final CUTLASS memcheck: **0 errors**. Racecheck: **0 errors / 0 warnings**. The unchanged CuTe backend retains its earlier zero-error sanitizer results.
- Every timed INT8 candidate passes sampled independent CPU references and full-output finiteness checks. The final auto, CUDA and CUTLASS APIs also pass per-shape Graph zero/restore replay checks.

Memcheck uses `--target-processes application-only --report-api-errors explicit`. The default extended-API mode reports an optional TileIR compiler initialization failure at `cudaGetDeviceCount` in this container; these native kernels use NVCC cubins. Explicit CUDA API and kernel memory errors remain checked. The [earlier extended diagnostic](../cute_v1/memcheck_extended_api.log) is retained.

## Independent API performance comparison

Offline tuning uses 3 rounds, 20 ms captured batches and 300 ms untimed warmup per candidate. For the 16 previously measured shapes, the unchanged CuTe/Triton implementations reuse their prior winning configurations as candidates and **remeasure** them against four legacy and two new CUTLASS candidates. The three added Cosmos shapes search eight Triton, up to 26 CUDA and six CUTLASS candidates. The fallback-table hash and all measured candidates are recorded in `tuning.json`; the previous complete search is retained under `cute_v1/`.

The table below is a fresh **5-round** comparison using 20 ms captured batches and 1500 ms warmup per implementation, with rotating execution order. The installed Triton helper replays each batch ten times; the median of round medians is reported. Auto is timed through the actual public API after installing the new dispatch table.

`Auto` names the backend and configuration selected by the offline dispatch table. Thor selection includes the new CUTLASS pipeline alongside CUDA/CuTe and Triton. `CuTe` is the separately forced `backend="cuda"` implementation; `Tuned Triton` is `backend="triton"`; `CUTLASS` is `backend="cutlass"`. Auto and the selected forced backend are measured separately, so their medians can differ.

Timing includes prequantized GEMM, group scaling and BF16 conversion. It excludes input quantization, compilation, tuning, warmup and host dispatch. Descriptor initialization is invoked inside each captured API call; host-only allocation and initialization work is outside the GPU timer. Repeated operands are used without an L2 flush. BF16 uses `torch.mm` on the original BF16 inputs and computes the corresponding unquantized operation. This is not Cosmos's historical cuBLASLt benchmark or an end-to-end inference measurement.

| M × N × K | Auto | Auto µs | CuTe µs | Tuned Triton µs | CUTLASS µs | BF16 µs | BF16 / Auto |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 × 1024 | cutlass #32 | 16.75 | 36.93 | 35.67 | 16.74 | 18.49 | 1.10× |
| 2048 × 2048 × 2048 | cutlass #32 | 77.39 | 198.66 | 224.62 | 77.48 | 126.73 | 1.64× |
| 3072 × 3072 × 3072 | cutlass #32 | 276.23 | 643.61 | 725.51 | 275.80 | 501.19 | 1.81× |
| 4096 × 4096 × 4096 | cutlass #33 | 658.55 | 1656.35 | 1747.69 | 658.60 | 1106.34 | 1.68× |
| 6144 × 6144 × 6144 | cutlass #33 | 2208.98 | 5278.02 | 5853.32 | 2209.89 | 4083.30 | 1.85× |
| 8192 × 8192 × 8192 | cutlass #33 | 5336.16 | 11984.64 | 13908.82 | 5360.91 | 10094.24 | 1.89× |
| 12288 × 12288 × 12288 | cutlass #32 | 18408.40 | 39956.79 | 46133.63 | 18431.15 | 35123.86 | 1.91× |
| 16384 × 16384 × 16384 | cutlass #32 | 65056.66 | 93486.31 | 108729.01 | 68107.60 | 90548.91 | 1.39× |
| 256 × 1024 × 1536 | cutlass #32 | 10.84 | 13.42 | 15.01 | 10.88 | 9.85 | 0.91× |
| 256 × 1536 × 1024 | cutlass #32 | 9.35 | 16.76 | 14.81 | 9.36 | 8.82 | 0.94× |
| 256 × 2048 × 1536 | cutlass #32 | 11.64 | 25.88 | 27.58 | 11.70 | 18.11 | 1.56× |
| 1024 × 1024 × 1536 | cutlass #32 | 20.71 | 47.78 | 49.69 | 20.71 | 27.41 | 1.32× |
| 1024 × 1536 × 1024 | cutlass #32 | 23.27 | 48.95 | 49.19 | 23.29 | 31.92 | 1.37× |
| 1024 × 4096 × 1536 | cutlass #32 | 62.91 | 160.95 | 175.53 | 62.91 | 112.98 | 1.80× |
| 2048 × 2048 × 1536 | cutlass #32 | 64.05 | 161.10 | 176.18 | 64.07 | 121.18 | 1.89× |
| 2048 × 151936 × 1536 | cutlass #32 | 4723.10 | 13966.29 | 13346.58 | 4730.70 | 9541.13 | 2.02× |
| 512 × 3072 × 1536 | cutlass #32 | 29.24 | 63.57 | 67.59 | 29.24 | 46.76 | 1.60× |
| 4096 × 3072 × 1536 | cutlass #32 | 175.69 | 472.89 | 510.06 | 175.52 | 372.91 | 2.12× |
| 1536 × 1024 × 4096 | cutlass #32 | 57.56 | 138.20 | 164.53 | 57.57 | 133.10 | 2.31× |

The CSV also records original portable Triton, throughputs, ratios and every round. The JSON retains configurations, correctness checks, software versions and source/build/library hashes. Automatic native selection requires a measured advantage of at least 5% over the candidates in the tuning run. Unmeasured shapes use portable Triton in auto mode; explicit CUTLASS uses the new Thor default when compatible. Missing/stale native builds, unsupported layouts, other group sizes and other output types retain compatible fallbacks. These measurements establish neither universal speedups nor model accuracy.

Small projections can remain slower than BF16. For example, M256/N1024 provides only four 256×256 output tiles, each using two SMs, on this 20-SM GPU. Limited parallel work and short K are likely to make the fixed pipeline and group-promotion overhead harder to amortize.

## Reproduce

From the repository root with CUTLASS commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`:

```bash
python -m pip install -e ".[runtime,test]"
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_110
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
python -m pytest -q
compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python tests/sanitize_cutlass.py
compute-sanitizer --tool racecheck --target-processes application-only --error-exitcode 1 python tests/sanitize_cutlass.py
cp benchmarks/results/thor/cute_v1/dispatch.json src/grain_gemm/kernels/configs/thor_sm110_g256.json
python benchmarks/bench_thor.py --tune --reuse-fallbacks
python benchmarks/bench_thor.py --rounds 5 --rep-ms 20 --warmup-ms 1500
```

Use `--tune` without `--reuse-fallbacks` to search the full candidate set for every shape. Install the historical fallback table to reproduce this exact candidate set. In this container commands were run with `LD_LIBRARY_PATH='' PYTHONPATH=src`. Native binaries are built locally and excluded from distributions; source, shadow headers, helpers and dispatch data are included in the wheel. Restart Python after rebuilding a loaded library.

[Complete tuning](tuning.json) · [Independent comparison](comparison.json) · [Per-shape CSV](comparison.csv) · [Full tests](pytest.log) · [Memcheck](cutlass_memcheck.log) · [Racecheck](cutlass_racecheck.log)
