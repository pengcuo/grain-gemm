# Thor SM110 validation and performance

Validated on NVIDIA Thor, capability 11.0, 20 SMs. PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102, Triton 3.6.0, CUDA runtime 13.2; NVCC 13.2, target `sm_110a`, driver 620.54. Clocks and power policy were not changed.

The final public `backend="auto"` API is faster than PyTorch BF16 on **17/18 measured shapes**. Per-shape BF16/auto ratios range from **0.94× to 2.25×**; peak effective INT8 throughput is **243.10 TOPS**. These are individual shape results, not an average.

## Implementation and compatibility

The independent `backend="cuda"` CuTe implementation now has SM110-only warp-specialized candidates 26–29. Configs 26/27 use a persistent two-SM 256×256×256 tile with three/two operand stages and 48/216 register redistribution; configs 28/29 use one-SM 128×128/128×256 tiles. All use 384 threads, four producer warps, eight promotion warps, two INT32 TMEM buffers, three-fragment TMEM prefetch and vector scale/output accesses. They implement the pipeline using CuTe primitives and never invoke the separate CUTLASS GEMM backend. The scale-promotion and producer/consumer design follows the same ideas as CUTLASS #32.

Compared with the [previous CuTe measurements](https://github.com/pengcuo/grain-gemm/blob/560a964bf42b90c2d8640dd77ab5d119ab7f11f7/benchmarks/results/thor/cutlass_v2/comparison.json), the current forced CuTe latency improves by **1.39×–3.17×** across the 18 measured shapes. These before/after runs were recorded separately. The fresh same-run CUTLASS/CuTe latency ratio ranges from **0.82× to 1.13×**; the table below gives every shape.

CUTLASS configs 32/33 adapt the Thor G256 pipeline from `nvidia-cosmos/cosmos-int8-gemm`, commit `8977529aa23f70f02bfc36559599f00d59792996`. A two-SM 256×256×256 tile combines TMA INT8 loads, UMMA/TMEM, three fragment buffers, eight promotion warps, vector shared scale loads and 48/216/216 register redistribution. Config 32 retains the structured `EpilogueTileAuto`; config 33 is an explicit 128×16 candidate. The production builder enforces the Cosmos release flag `-DNDEBUG`.

The public `int8_gemm` and `get_kernel_config` APIs match GB10. A is INT8 row-major `[M,K]`, and logical B `[K,N]` is a transposed contiguous `[N,K]` tensor. Scales remain contiguous FP32 `scale_a[M,K/256]` and `scale_b[K/256,N]`. The Cosmos activation-scale layout is specialized to K-major; the API performs no transpose, packing, quantization or hidden copy. Each complete G256 INT32 dot contributes through `fmaf(float(partial), scale_a * scale_b, total)` to the FP32 sum, then BF16 conversion occurs once. Experimental biased and separable-scale modes are disabled.

Kernel parameters and descriptors are rebuilt for each invocation. The independent two-SM CuTe scheduler uses a static persistent grid, with barrier sequence numbers carried across output tiles. Every scale producer publishes its own completed shared stores, and every consumer releases the scale stage after reading it. Two-SM producers wait for their asynchronous scale copies before publication. The separate CUTLASS CLC scheduler requires zero scratch bytes; the ABI supports caller-owned scratch through PyTorch if a configuration requires it. CUDA Graph and concurrent streams share no cached native handle or pointer-dependent state. Four-byte scale offsets remain valid: auto falls back from the vector CUTLASS candidate to its CUDA/Triton selection; explicit CUTLASS uses its legacy scalar-compatible collective. The new CuTe candidates delegate unaligned weight-scale pointers to warp-MMA config 0. The aligned-input performance table does not measure these offset fallbacks.

GB10 sources, both GB10 tuning tables, all existing configuration IDs and the legacy C ABI remain unchanged.

## Correctness

- Full suite after final dispatch installation: **937 passed, 13 skipped**.
- Independent CPU INT32 group-dot references cover all 30 Thor CUDA and 34 CUTLASS configurations. New pipeline tests include K256/512/768/4352, minimum/partial tiles, signed/zero/distinct scales, INT8 extrema, exact `2^-36` cancellation and a `-2^-23` residual requiring fused multiply-add.
- Operand/scale offsets, CUDA Graph replay with updated operands and scales, non-default streams, concurrent streams and missing-library fallbacks pass.
- Final CuTe memcheck: **0 errors**. Racecheck: **0 errors / 0 warnings**, including persistent output-tile reuse with odd group counts. The unchanged CUTLASS backend retains its previous zero-error sanitizer results.
- Every timed INT8 candidate passes sampled independent CPU references and full-output finiteness checks. The final auto, CUDA and CUTLASS APIs also pass per-shape Graph zero/restore replay checks.

Memcheck uses `--target-processes application-only --report-api-errors explicit`. The default extended-API mode reports an optional TileIR compiler initialization failure at `cudaGetDeviceCount` in this container; these native kernels use NVCC cubins. Explicit CUDA API and kernel memory errors remain checked.

## Independent API performance comparison

Offline tuning uses 3 rounds, 20 ms captured batches and 300 ms untimed warmup per candidate. The completed offline tuning remeasured 19 shapes before the user excluded 16384 cubed due to device instability. It **remeasured** the prior CuTe/Triton winners, compatible new CuTe candidates and six CUTLASS candidates (7/15/23/31/32/33): 11–12 candidates per shape, 227 measurements in total. The 151936-column shape also tests the two-SM CuTe half-tile path. The fallback-table hash and all measured candidates are recorded in `tuning.json`. Earlier tables, measurements and diagnostic snapshots remain available in Git history at commit `560a964bf42b90c2d8640dd77ab5d119ab7f11f7`.

The final comparison and the benchmark default now **exclude 16384×16384×16384** and cover only 18 shapes. The retained tuning data includes its earlier result; reproduction commands use the current 18-shape default.

The table below is a fresh **5-round** comparison using 20 ms captured batches and 1500 ms warmup per implementation, with rotating execution order. The installed Triton helper replays each batch ten times; the median of round medians is reported. Auto is timed through the actual public API after installing the new dispatch table.

`Auto` names the backend and configuration selected by the offline dispatch table. Thor selection includes the new CUTLASS pipeline alongside CUDA/CuTe and Triton. `CuTe` is the separately forced `backend="cuda"` implementation; `Tuned Triton` is `backend="triton"`; `CUTLASS` is `backend="cutlass"`. Auto and the selected forced backend are measured separately, so their medians can differ.

Timing includes prequantized GEMM, group scaling and BF16 conversion. It excludes input quantization, compilation, tuning, warmup and host dispatch. Descriptor initialization is invoked inside each captured API call; host-only allocation and initialization work is outside the GPU timer. Repeated operands are used without an L2 flush. BF16 uses `torch.mm` on the original BF16 inputs and computes the corresponding unquantized operation. This is not Cosmos's historical cuBLASLt benchmark or an end-to-end inference measurement.

The CuTe **TFLOPS (equiv.)** column is `2MNK / (time_seconds × 10^12)`. For INT8, the standard unit is TOPS; the equivalent TFLOPS number uses the same operation-count convention.

| M × N × K | Auto | Auto µs | CuTe µs | CuTe TFLOPS (equiv.) | Tuned Triton µs | CUTLASS µs | BF16 µs | BF16 / Auto |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 × 1024 × 1024 | cutlass #32 | 16.70 | 18.34 | 117.12 | 35.68 | 16.64 | 17.66 | 1.06× |
| 2048 × 2048 × 2048 | cuda #27 | 79.69 | 79.70 | 215.56 | 224.86 | 77.38 | 121.62 | 1.53× |
| 3072 × 3072 × 3072 | cuda #26 | 243.62 | 242.46 | 239.15 | 726.04 | 259.00 | 488.92 | 2.01× |
| 4096 × 4096 × 4096 | cuda #26 | 574.10 | 576.01 | 238.61 | 1748.92 | 594.73 | 1137.24 | 1.98× |
| 6144 × 6144 × 6144 | cuda #27 | 1908.05 | 1912.45 | 242.55 | 5854.56 | 2034.88 | 3856.24 | 2.02× |
| 8192 × 8192 × 8192 | cuda #26 | 4672.41 | 4687.66 | 234.55 | 13908.48 | 4917.27 | 9446.99 | 2.02× |
| 12288 × 12288 × 12288 | cuda #26 | 15802.02 | 15976.19 | 232.27 | 46261.04 | 17740.03 | 32071.87 | 2.03× |
| 256 × 1024 × 1536 | cuda #28 | 9.73 | 9.62 | 83.68 | 15.13 | 10.84 | 9.84 | 1.01× |
| 256 × 1536 × 1024 | cutlass #32 | 9.45 | 11.47 | 70.21 | 15.10 | 9.36 | 8.89 | 0.94× |
| 256 × 2048 × 1536 | cutlass #32 | 11.77 | 14.01 | 114.96 | 27.91 | 11.87 | 18.02 | 1.53× |
| 1024 × 1024 × 1536 | cutlass #32 | 20.85 | 22.87 | 140.83 | 50.26 | 20.95 | 25.92 | 1.24× |
| 1024 × 1536 × 1024 | cutlass #32 | 23.45 | 24.71 | 130.36 | 49.63 | 23.47 | 32.21 | 1.37× |
| 1024 × 4096 × 1536 | cutlass #32 | 62.67 | 66.39 | 194.09 | 175.95 | 62.68 | 111.25 | 1.78× |
| 2048 × 2048 × 1536 | cuda #27 | 66.68 | 66.65 | 193.31 | 176.06 | 63.86 | 106.84 | 1.60× |
| 2048 × 151936 × 1536 | cuda #27 | 4409.02 | 4408.84 | 216.81 | 13340.35 | 4717.63 | 8469.75 | 1.92× |
| 512 × 3072 × 1536 | cutlass #32 | 29.18 | 31.61 | 152.87 | 67.66 | 29.16 | 42.63 | 1.46× |
| 4096 × 3072 × 1536 | cutlass #32 | 173.87 | 188.25 | 205.34 | 509.66 | 175.16 | 351.73 | 2.02× |
| 1536 × 1024 × 4096 | cutlass #32 | 57.58 | 61.75 | 208.66 | 164.52 | 57.59 | 129.32 | 2.25× |

The CSV also records original portable Triton, throughputs, ratios and every round. The JSON retains configurations, correctness checks, software versions and source/build/library hashes. Auto uses the offline tuning results: CuTe must be at least 5% faster than Triton to replace it; CUTLASS must be at least 5% faster than the faster of CuTe and Triton to replace that selection. Otherwise it retains the CuTe/Triton selection, so Auto is not the literal minimum of all columns. Unmeasured shapes use portable Triton in auto mode; explicit CUTLASS uses the new Thor default when compatible. Missing/stale native builds, unsupported layouts, other group sizes and other output types retain compatible fallbacks. These measurements establish neither universal speedups nor model accuracy.

Small projections provide less parallel work and shorter K to amortize pipeline and group-promotion overhead. The measured one-SM and two-SM candidates allow the dispatch table to choose different tile sizes for these shapes; their results are listed above.

## Reproduce

From the repository root with CUTLASS commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`:

```bash
python -m pip install -e ".[runtime,test]"
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_110
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
python -m pytest -q
compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python tests/sanitize_cutlass.py
compute-sanitizer --tool racecheck --target-processes application-only --error-exitcode 1 python tests/sanitize_cutlass.py
compute-sanitizer --tool memcheck --target-processes application-only --report-api-errors explicit --error-exitcode 1 python tests/sanitize_kernels.py
compute-sanitizer --tool racecheck --target-processes application-only --error-exitcode 1 python tests/sanitize_kernels.py
python benchmarks/bench_thor.py --tune --reuse-fallbacks
python benchmarks/bench_thor.py --rounds 5 --rep-ms 20 --warmup-ms 1500
```

The tuning command remeasures the installed dispatch table's CUDA/Triton winners alongside compatible new CuTe and CUTLASS candidates. Candidate counts can differ from the historical tuning run. Use `--tune` without `--reuse-fallbacks` to search the full candidate set for every shape. In this container commands were run with `LD_LIBRARY_PATH='' PYTHONPATH=src`. Native binaries are built locally and excluded from distributions; source, shadow headers, helpers and dispatch data are included in the wheel. Restart Python after rebuilding a loaded library.

[Complete tuning](tuning.json) · [Independent comparison](comparison.json) · [Per-shape CSV](comparison.csv)

## Complete INT8 MoE FFN at reduced frequency

The separate [MoE FFN optimization report](moe_ffn_optimized.md) compares the
previous INT8 implementation, optimized INT8, upstream vLLM BF16 and Cosmos
compiled BF16 in a fresh paired run. **GPU frequency was reduced to 1.2 GHz
(actual 1197 MHz).** At 4096 tokens, the full INT8 FFN reaches 64.30 effective
TOPS: 2.405 ms versus 4.297 ms before optimization and 3.286 ms for vLLM BF16.
These full-FFN measurements use a different scope and frequency from the
historical single-GEMM results above.
