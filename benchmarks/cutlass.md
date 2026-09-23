# CUTLASS collective comparison

**GB10-only experiment: SM121 / compute capability 12.1.** The performance
results, tuned configurations and CuTe/CUTLASS rankings on this page apply only
to NVIDIA GB10. Thor / Jetson T5000 is SM110 / compute capability 11.0 and needs
its own implementation, tuning and validation. This scope applies to this
experiment, not GrainGEMM's future multi-architecture support.

The optional CUTLASS backend provides the same prequantized INT8 G256 operation
as the direct CuTe backend: each complete K256 group accumulates in INT32,
then contributes to an FP32 running sum through row/column scaling and FMA.
The output is converted to BF16 once. The validated target is GB10 / SM121;
an experimental SM120 build is also available through explicit backend
selection. Both require the
[native input layout](../docs/kernel_design.md#dispatch-and-limits).

It composes a custom SM80-style collective with CUTLASS's `GemmUniversal`,
`GemmUniversalAdapter`, and selectable output epilogues. A local kernel
specialization supplies FP32 accumulator fragments and scale parameters.
This backend does not call the existing direct CuTe kernel.
Its installed source lives under
[`csrc/sm12x/cutlass/`](../src/grain_gemm/kernels/csrc/sm12x/cutlass);
the architecture directory is shared by SM121 and experimental SM120 builds,
while measured launch tables remain specific to GB10.

Published data are indexed in [CUTLASS experiment results](results/cutlass/README.md).
The isolated K-lower-bound, output-addressing and synchronization variants live
under [experiments/cutlass](../experiments/cutlass/README.md), with their own
build and paired-measurement entry points. They do not replace the installed
backend or participate in `auto` dispatch.

## Build and use

Use the same CUTLASS checkout as the CuTe build, pinned to
`098de2a652cf8f00fd70b2df54051c7eccbb855a`:

```bash
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_121
```

This writes `libgrain_cutlass.so` and `cutlass_build.json` beside the existing
CuTe library. It does not replace `libgrain_cuda.so`. The build requires CUDA
13.0 or newer and a C++17 host compiler. No PyTorch C++ extension is involved.
Both installed-backend build entry points default to `--arch sm_121` and also
accept the experimental `--arch sm_120` target:

```bash
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_120
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_120
```

RTX 50-series runtime correctness and performance have not been validated.
SM120 requires explicit `backend="cuda"` or `backend="cutlass"` selection and
uses fixed compatible defaults rather than GB10 tuning tables; `auto` stays on
Triton for SM120. Other build targets are rejected before invoking the compiler.
Each backend installs a single target, so rebuilding replaces that backend's
previous library. Runtime checks that its GPU target matches the device.

Builds can run without a GPU, but the resulting library must match the execution
machine's CPU architecture and host ABI as well as the GPU target. For example,
a library built with a GB10 ARM64 host compiler cannot be loaded in an x86-64
desktop process. The isolated experiment builders under `experiments/cutlass/`
still accept only SM121, and their GPU entry points still require GB10.

```python
c = int8_gemm(a, b, scale_a, scale_b,
              group_size=256, output_dtype=torch.bfloat16, backend="cutlass")
```

`backend="cuda"` selects direct CuTe; `backend="cutlass"` selects the new
collective. The existing `auto` policy is retained. Explicit CUTLASS selection
reports an error if its native library or input constraints are unavailable.
On GB10, the dispatch table stores separate measured CUTLASS configurations for the
three M256 shapes and all sixteen square sizes below. These entries are used
only by `backend="cutlass"`; each has a `cutlass_measurement_sha256` identifying
its comparison record. Unmeasured shapes use a fixed compatible configuration.

`gb10_sm121_g256_cutlass.json` additionally stores the independently tuned choices
for 18 model projection shapes at M=1024 and M=2048. This separate table takes
precedence for explicit CUTLASS calls and does not alter `auto`, `cuda`, or
`triton` selection. Inspect a choice with
`get_kernel_config(a, b, scale_a=scale_a, scale_b=scale_b, group_size=256,
output_dtype=torch.bfloat16, backend="cutlass")`.

## Compare performance

Build both native backends first, then run:

```bash
python benchmarks/compare_cutlass.py --output-dir benchmarks/results/cutlass_g256
```

GPU measurement and validation entry points check both that the device name
contains `GB10` and that compute capability is `(12, 1)`, rejecting other GPUs.
CPU-only rendering and source/data audits have no GPU requirement.

The default sweep contains the three M256 shapes `(256,1024,1536)`,
`(256,1536,1024)`, `(256,2048,1536)`, plus M=N=K from 1024 to 16384 in steps
of 1024. Each shape runs in a separate process and produces a JSON checkpoint.
The directory also contains a Markdown table and CSV with explicit M/N/K,
latency, selected configurations, and throughput. No plot is required.
Completed checkpoints are reused only when source hashes and the runtime
environment match. Use a fresh output directory after source, configuration,
candidate, or toolchain changes; rendering rejects mixed measurement sources.

For a single shape or to render existing checkpoints:

```bash
python benchmarks/compare_cutlass.py --output-dir benchmarks/results/cutlass_g256 --case 256 1024 1536
python benchmarks/compare_cutlass.py --output-dir benchmarks/results/cutlass_g256 --render-only
```

The candidate sets are 32 CUTLASS configurations, seventeen CuTe
configurations, and four Triton configurations. CUTLASS repeats the same eight
tile/pipeline choices in four families:

| IDs | Scale loading | Output epilogue |
|---|---|---|
| 0–7 | Register prefetch | Upstream FP32 shared-memory exchange, 128-bit BF16 stores |
| 8–15 | Register prefetch | Convert to BF16 before shared-memory exchange, 128-bit stores |
| 16–23 | Asynchronous shared-memory staging | Direct adjacent BF16 pair stores |
| 24–31 | Register prefetch | Direct adjacent BF16 pair stores |

IDs 0–7 retain their original meaning and measured dispatch entries. The
additional families expose the output-exchange and scale-loading changes
separately; ID modulo eight identifies the common tile/pipeline choice. A
family need not be faster for every shape. The report retains every candidate's
configuration and tuning measurements, not just the selected results.

All backends receive the same INT8 inputs and FP32 scales. Each backend is
tuned independently, then its selected configuration is checked and timed
again after replacing inputs with a fresh fixed seed. Timing uses CUDA Graphs,
preallocated output, warm caches, rotating backend order and five final rounds.
It includes group scaling and BF16 output, but excludes input quantization,
compilation, tuning, allocation and Python/host launch overhead.

INT8 throughput is reported as TOPS. The table's equivalent TFLOPS column has
the same numerical value, using `2MNK / time`; it is not measured floating-point
instruction throughput. Comparisons characterize these implementations and
candidate sets, not an inherent performance ranking of CUTLASS and CuTe.

## Correctness and synchronization

```bash
python -m pytest -q tests/test_cutlass_gemm.py
compute-sanitizer --tool memcheck --error-exitcode 1 python tests/sanitize_cutlass.py
compute-sanitizer --tool racecheck --error-exitcode 1 python tests/sanitize_cutlass.py
```

Tests cover all configurations, short and long K, nonuniform signed
scales, INT8 extrema, FMA cancellation, streams and graph replay. Benchmark
checks use an independent CPU group reference at up to 32×32 sampled outputs,
plus finite-output and zero/restore graph replay checks.
