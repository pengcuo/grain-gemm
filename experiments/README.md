# Reproducible experiments

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

CUDA binaries, toolchain checkouts and installed dependencies are not bundled.
Historical source snapshots retain their own defaults and licenses; the main
API now defaults to BF16. See [third-party notices](../THIRD_PARTY_NOTICES.md).
