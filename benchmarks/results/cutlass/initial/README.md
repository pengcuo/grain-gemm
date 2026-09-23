# Initial 8-configuration CUTLASS comparison

This archived run contains 19 measured shapes on GB10, INT8 G256 inputs, FP32 scales/accumulation and BF16 outputs. See [all timings and ranges](report.md) and [CSV](results.csv). Integer TOPS uses `2MNK/time`; any historical equivalent-TFLOPS column has the same numerical value. Quantization and host dispatch are excluded.

The raw JSON/CSV and original report are unchanged. They describe the source/candidate sets at measurement time, including the original eight CUTLASS configurations; they do not benchmark the later 32-configuration implementation. See [subsequent optimization](../optimized/README.md) and [fixed-configuration ablations](../residual-gap/README.md). Current dispatch defaults may differ.

Historical source snapshots are in [experiments/cutlass/history/initial](../../../../experiments/cutlass/history/initial/). Native binaries and third-party dependencies are not distributed. File hashes and original source locations are retained in [archive manifest](archive_manifest.json); absolute paths in raw provenance are historical records, not instructions to use that directory.
