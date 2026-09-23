# LLM prefill experiment

[Historical completed measurements](../../benchmarks/results/llm-prefill/README.md)

`benchmark.py` reruns the 51-shape workload against the **currently installed GrainGEMM package**, with separate forced-backend defaults and local tuning. It records the installed sources, dispatch-table hash and native binary hash. Current default selections may differ from the historical archive, especially at M=256. Never interpret a new run as an exact reproduction of the archived defaults.

Use a GB10 with the CuTe native backend built and install the package with runtime dependencies. From the repository root:

```bash
python experiments/prefill/benchmark.py --output /tmp/grain-prefill-new/results.json
python experiments/prefill/validate_results.py /tmp/grain-prefill-new/results.json --validate-only
```

`--limit 1` is a short smoke run. `--max-new-cases 1` saves one new case and `--resume` continues an incomplete run after checking software, source and timing compatibility. New runs refuse to replace existing results unless resuming and refuse to write into `benchmarks/results/`.

The historical rendering/validation script is preserved as `validate_results.py`; use **`--validate-only`** to validate the archived data without rewriting the publication report:

```bash
python experiments/prefill/validate_results.py benchmarks/results/llm-prefill/results.json --validate-only
```

`historical_benchmark.py` is an unchanged provenance snapshot. Its absolute paths and original dependency environment are historical, so it is not the portable entrypoint. No GPU binaries or third-party dependencies are bundled. [manifest.json](manifest.json) identifies both historical source files and the portable path/output changes.
