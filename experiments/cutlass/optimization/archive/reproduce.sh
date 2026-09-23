#!/usr/bin/env bash
set -euo pipefail

experiment_dir=/home/pengcuo/spark/experiments/grain-cutlass-optimized
export OMP_NUM_THREADS=4
# Use the archived package and binaries to retain the measured source version.
export PYTHONPATH="$experiment_dir/measurement_sources":/home/pengcuo/spark/experiments/grain-m256-root-cause/.test-deps:/home/pengcuo/spark/experiments/grain-llm-prefill-g256/.deps
export CPATH=/home/pengcuo/spark/experiments/grain-llm-prefill-g256/.python-dev/usr/include/python3.12:/home/pengcuo/spark/experiments/grain-llm-prefill-g256/.python-dev/usr/include

# Supply a fresh output directory to repeat timings instead of reusing checkpoints.
result_dir=${1:-"$experiment_dir/results"}
/home/pengcuo/spark/venv/bin/python -u "$experiment_dir/compare_optimized.py" --output-dir "$result_dir"
