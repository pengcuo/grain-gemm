# CUTLASS/CuTe residual-gap experiments

Fixed-configuration ablations on GB10, INT8 G256, FP32 scales/accumulation and BF16 output. All timings exclude quantization.

- [Investigation and TOPS table](report.md) · [run3 CSV](results_run3.csv)
- [M=4096 results with TOPS](paired_m4096/report.md) · [M=4096 CSV](paired_m4096/results.csv)
- Raw cases: [run1](paired_run1/), [run2](paired_run2/), [run3](paired_run3/), [M=4096](paired_m4096/)
- [Historical paired audit](paired-audit.json) · [M=4096 audit](paired-m4096-audit.json) · [Static evidence](static-report.md)
- [Variant memcheck](memcheck_guard_ungrouped.log) · [Variant racecheck](racecheck_guard_ungrouped.log) · [Profiler permission result](ncu_probe.log)
- [Build and reproduce](../../../../experiments/cutlass/README.md) · [Source manifest](../source_manifest.json) · [Offline verification](../README.md)

The 19 raw cases contain 115 measured paths and 20,700 event-timed graph replay samples. Each path has 60 rounds of three replays; all paths for a shape use identical graph-node counts and inputs. Orders rotate and reverse. Runs 1/3 and M=4096 are position-balanced; run2 has seven paths and 60 rounds, so its first 56 rounds provide the strictly balanced sensitivity check. Ratios use median paired per-round ratios, which may differ from the ratio of aggregate medians.

The K-minimum source guard closes the tested 128×128 projection gap; it is not a universal performance improvement. Removing tail synchronization explains only part of the LM-head gap. No hardware counter data are available (`ERR_NVGPUCTRPERM`). Timings come from one device with unlocked clocks, warm caches and the same input seed. Bootstrap intervals describe these runs, not independent machines; serial correlation remains. Timed matrices receive sampled reference and full-output finite checks, not full-matrix reference comparisons. Dedicated small-shape checks and sanitizers are separate evidence.

Original JSON/CSV, audits and logs are byte-identical to local records. Absolute paths/hashes are historical provenance; binaries are not included. `*.summary.json` files omit instruction arrays and are explicitly derived. Original human-readable reports are preserved as `.original.txt`; Markdown reports update links and publication wording without changing measurements.
