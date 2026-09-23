# CUTLASS optimization: 18 projection shapes

GB10, G256, INT8 inputs and BF16 outputs. M is 1024 or 2048. The experiment retunes and remeasures the original 8-configuration CUTLASS library, optimized 32-configuration CUTLASS library, CuTe, and Triton within the same run.

- [Results, TOPS and interpretation](report.md)
- [Full family ablations and round ranges](results/report.md) · [Original CSV](results/results.csv)
- [Historical statistics/source audit](audit_snapshot.json) · [Tests](pytest_integrated.log) · [memcheck](memcheck.log) · [racecheck](racecheck.log)
- [Source manifest](../source_manifest.json) · [Offline public-data verification](../README.md)

`results/case_*.json` contains 18 raw cases, 1188 tuning records and 144 final records. Timings measure prequantized GEMM including scaling/output conversion, excluding quantization, compilation, tuning and host dispatch. Repeated inputs and warm caches are used. Every candidate has sampled reference checks; sampled checks are not full-matrix verification.

INT8 throughput is TOPS = 2MNK / time / 10^12. Original CSV fields named `equivalent_tflops` are preserved unchanged and have the same numeric value.

Historical absolute paths and binary hashes remain provenance. No binary or third-party checkout is distributed; archived binary-audit success records do not mean an independent reader can revalidate the unavailable binaries. Rebuild and rerun this phase with the [optimization reproduction entry](../../../../experiments/cutlass/optimization/README.md). It retains the measured source snapshot and original tuning protocol; rebuilding does not guarantee bit-identical binaries across toolchain versions.
