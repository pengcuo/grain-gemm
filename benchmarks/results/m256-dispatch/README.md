# GB10 M256 dispatch investigation

[Root-cause report](report.md) · [All 69 shape/config rows](results.csv) · [Static audit](static-audit.md) · [Experiment scripts](../../../experiments/m256-dispatch/README.md)

This historical investigation contains **14 cases and 69 shape/config combinations**: three M256 shapes, six fixed-M N sweeps and five fixed-N M sweeps. It supports grid size and CTA residency boundaries as the principal structural explanation for the two slow default CuTe cases. It does not assign cycle-level stall causes; Nsight Compute hardware counters were unavailable.

The resulting **exact-shape** selections are now integrated into auto dispatch for GB10/G256/BF16/native layout:

| M | N | K | CuTe config | Tile |
|---:|---:|---:|---:|---|
| 256 | 1024 | 1536 | 3 | 128 × 64 |
| 256 | 1536 | 1024 | 5 | 64 × 128 |
| 256 | 2048 | 1536 | 7 | 128 × 128 |

The report’s “default” timings describe pre-integration selection; they are not measurements of the current auto policy. Every run explicitly uses BF16 output. The API’s later BF16 default change does not alter these raw results.

Each JSON contains all 5 × 9 graph replay timings per candidate, plus source/native hashes and sampled correctness results. CSV throughput uses the equivalent-TFLOPS label; its numerical value is INT8 TOPS (`2MNK/time`). [occupancy.json](occupancy.json) and [static-audit.json](static-audit.json) describe the exact historical native binary. They are not performance counters.

Raw data and compact resource records are byte-preserved; [manifest.json](manifest.json) lists their hashes. Reports have only publication links and historical-state wording updated. Large disassemblies, binary artifacts and local dependencies are excluded.
