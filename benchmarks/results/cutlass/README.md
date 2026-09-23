# CUTLASS experiment results

**All results here are scoped to NVIDIA GB10 (SM121, compute capability 12.1).**
Performance, tuned configurations and CuTe/CUTLASS rankings cannot be carried
over to Thor / Jetson T5000 (SM110, compute capability 11.0), which requires an
independent implementation, tuning and validation. GrainGEMM's multi-architecture
goal remains unchanged; this is the boundary of this experimental evidence.

Measured GB10 INT8 G256 GEMM comparisons. Inputs are prequantized, scales and accumulation are FP32, and outputs are BF16. Timings include group scaling and output conversion, but exclude quantization. INT8 throughput is reported as TOPS (`2MNK / time / 10^12`).

| Experiment | Scope | Results |
|---|---|---|
| Initial CUTLASS comparison | 19 baseline shapes | [Report](initial/report.md) |
| Model projections before optimization | 18 M=1024/2048 shapes | [Report](model-projections/report.md) |
| 32-configuration optimization | 18 projection shapes, M=1024/2048; original CUTLASS, optimized CUTLASS, CuTe, Triton | [Report](optimized/report.md) · [CSV](optimized/results/results.csv) |
| Residual-gap investigation | 5 fixed shapes × 3 paired runs, independent source ablations | [Report](residual-gap/report.md) · [CSV](residual-gap/results_run3.csv) |
| M=4096 extension | 4 fixed projection shapes; CuTe and CUTLASS variants | [Report with TOPS](residual-gap/paired_m4096/report.md) · [CSV](residual-gap/paired_m4096/results.csv) |

[Experimental source and reproduction](../../../experiments/cutlass/README.md) live outside the runtime implementation. A shape-specific result does not establish a universally faster backend. These experiments do not measure a whole model or include activation quantization.

Portable GPU measurement and validation entry points require both a device name
containing `GB10` and compute capability `(12, 1)`. They reject other GPUs;
CPU-only rendering and audits do not require a GPU. Historical reports and raw
measurements retain their original data.
Build entry points accept only `--arch sm_121`; other targets are rejected.
Cross-compilation requires the toolchain but does not require a GPU.

Raw JSON/CSV and evidence logs are preserved byte-for-byte. For the optimization and residual-gap stages, [source_manifest.json](source_manifest.json) records the original local paths, sizes and SHA256 hashes, and labels derived static summaries. Earlier stages retain their own source manifests. Absolute paths inside original files are historical provenance, not paths readers must recreate. Dynamic libraries, complete SASS dumps and the third-party CUTLASS checkout are excluded. Historical audits record checks performed at measurement time; the public archive can independently verify file integrity and recorded statistics, not the unavailable original binaries.

Run the CPU-only verification from any directory:

```bash
python /path/to/grain-gemm/benchmarks/results/cutlass/verify_archive.py
```

It checks archived hashes, raw-round reductions, paired ratios, published CSV throughput, and Markdown links without importing PyTorch or loading CUDA. [Publication audit](publication_audit.json) records the latest check. Original human-readable reports are also preserved as `.original.txt`; published Markdown updates links/publication wording and clarifies TOPS.
