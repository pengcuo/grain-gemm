# Reproducible experiments

**This published experimental batch targets NVIDIA GB10 only: SM121 / compute
capability 12.1.** Its performance results, tuned configurations and backend
rankings, including CuTe versus CUTLASS, are specific to GB10. Thor / Jetson
T5000 is SM110 / compute capability 11.0 and requires separate implementation,
tuning and validation. GrainGEMM can continue adding other architectures; these
experiments do not establish their support or performance.

This directory contains measurement tools and isolated implementation changes.
The installable library remains under `src/grain_gemm/`; experimental changes
are not automatically selected by its public API.

- [CUTLASS comparisons and source ablations](cutlass/README.md)
- [Original versus optimized CUTLASS, M=1024/2048](cutlass/optimization/README.md)
- [LLM prefill measurement archive](prefill/README.md)
- [M=256 dispatch diagnosis](m256-dispatch/README.md)

Read the [benchmark index](../benchmarks/README.md#published-gb10-experiments)
for measured results. Raw JSON and CSV preserve the original values and hashes.
Source archives retain historical build metadata and local paths as provenance;
portable entry points document how to run from a different checkout. Rebuilt
binaries can have different hashes and performance.

Portable GPU measurement and validation entry points check the device name for
`GB10` and require compute capability `(12, 1)`, rejecting other GPUs. CPU-only
report rendering and source/data audits require no GPU. The archived sources
and reports remain historical records; use the portable entry points for new
runs.
Native and experimental build entry points accept only `--arch sm_121` and
reject other targets. Cross-compilation itself does not require a GPU.

CUDA binaries, toolchain checkouts and installed dependencies are not bundled.
Historical source snapshots retain their own defaults and licenses; the main
API now defaults to BF16. See [third-party notices](../THIRD_PARTY_NOTICES.md).
