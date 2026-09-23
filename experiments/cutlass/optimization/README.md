# Original versus optimized CUTLASS: 18 projection shapes

**GB10-only experiment: SM121 / compute capability 12.1.** All performance,
tuned configurations and CuTe/CUTLASS rankings in this comparison apply only to
NVIDIA GB10. Thor / Jetson T5000 is SM110 / compute capability 11.0 and needs
independent implementation, tuning and validation. This experimental scope does
not limit GrainGEMM's goal of supporting additional architectures.

This experiment compares the original eight CUTLASS configurations with the
32-configuration implementation, CuTe and Triton at M=1024/2048. Each backend
and each CUTLASS family is tuned independently, then measured on fresh inputs.
See the [published results](../../../benchmarks/results/cutlass/optimized/README.md).

The portable GPU entry point checks both a device name containing `GB10` and
compute capability `(12, 1)`, and rejects other GPUs. CPU-only report rendering
and source/data audits do not require a GPU. Historical reports and raw
measurements remain unchanged.

Run from the repository root with CUDA-enabled PyTorch, Triton, CUDA 13.0 and
CUTLASS commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`:

```bash
python experiments/cutlass/optimization/build.py --cutlass-dir /path/to/cutlass
python experiments/cutlass/optimization/compare_optimized.py --output-dir /path/to/fresh-results
```

The build entry point accepts only `--arch sm_121`; other targets are rejected
during argument parsing. Cross-compilation requires the toolchain but no GPU.

For one shape, append `--case 1024 1536 1024`. The benchmark loads its archived
Python package and locally rebuilt libraries. It does not use or change the
current library's automatic dispatch. INT8 inputs are already quantized,
group scales and accumulation are FP32, and outputs are BF16. Timing excludes
quantization, tuning, compilation, allocation and host dispatch.

`before/` and `measurement_sources/` preserve the measured source files byte for
byte. Their `_native/*.json` files and `before_hashes.json` are historical build
records; the binaries themselves are not distributed. Rebuilding creates new
libraries and a separate `rebuild_manifest.json`. Compiler, path or environment
differences can change library hashes and timings, so historical binary hashes
are provenance, not a promise of byte-identical rebuilt libraries.

`archive/compare_optimized.py` is the original measurement script. The portable
copy adds the archived package to `sys.path` and enforces the GB10 device check;
its new script hash is recorded in new measurements. `tune_grain.py` preserves
all nine historical Triton candidates. `archive_manifest.json` records hashes
of the copied source.
The archived audit/reproduction scripts retain their original paths for
provenance and are not the portable entry points above.

New measurements require a fresh output directory. Do not mix historical
checkpoints with newly built code or rerender them as though their hashes matched
the new script. The original source and license notices are retained; see
[third-party notices](../../../THIRD_PARTY_NOTICES.md).
