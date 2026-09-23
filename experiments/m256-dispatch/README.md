# M256 grid and dispatch experiment

[Historical results and root-cause report](../../benchmarks/results/m256-dispatch/README.md)

The GPU entry point checks for an NVIDIA GB10 device and compute capability
12.1 (SM121) before native loading or measurement. These tile/resource choices
are not configurations for Thor or other GPU architectures.

The portable `measure.py` keeps the historical fixed CuTe configuration IDs and Triton tiles. It records the **current** API selections separately under `defaults`; those selections now include three measured M256 entries and are not the defaults in the archived JSON.

Requirements are GB10/SM121, the built CuTe backend, runtime dependencies and an occupancy record for the **same native binary SHA256**. The default occupancy input is the archived record. The script refuses to assign those resources to a different rebuild; a different binary requires a fresh matching occupancy record passed through `--occupancy`. The archive does not bundle the historical binary or a complete cubin extraction toolchain.

Run from the repository root into a fresh directory:

```bash
python experiments/m256-dispatch/measure.py --suite main --n 1024 --output-dir /tmp/grain-m256-new
python experiments/m256-dispatch/measure.py --suite main --n 1536 --k 1024 --output-dir /tmp/grain-m256-new
python experiments/m256-dispatch/measure.py --suite main --n 2048 --output-dir /tmp/grain-m256-new
```

For the fixed-M boundary sweep, use `--suite n_sweep --n N` for N=1408,1536,1664,2944,3072,3200. For the fixed-N sweep, use `--suite m_sweep --m M --n 1024` for M=128,384,512,768,896. Both retain K=1536. Together with the three main cases these reproduce the 14-case experimental design, subject to the recorded binary/toolchain conditions.

All variants use the same deterministic input prefixes, 256 nodes per graph, five rotating rounds and nine graph replays per round. Output is BF16 and quantization is excluded. New JSON refuses to overwrite an existing file or to write into `benchmarks/results/`.

`historical_measure.py` is an unchanged source snapshot with the original machine-local paths. It is archival evidence, not the portable entrypoint. [manifest.json](manifest.json) records the portable path/output/resource checks separately from the original source hash.
