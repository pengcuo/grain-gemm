#!/usr/bin/env python3
"""Verify frozen source hashes and their links to historical measurement JSON."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        help="Optional residual-gap results directory containing paired runs")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Archived source checksum differs: {name}")
    for path in (ROOT / "variants").glob("*/manifest.json"):
        variant = json.loads(path.read_text())
        for name, expected in variant["source_sha256"].items():
            if sha(path.parent / name) != expected:
                raise ValueError(f"Historical variant checksum differs: {path.parent.name}/{name}")
    records = 0
    if args.results_dir:
        for path in args.results_dir.expanduser().resolve().rglob("case_*.json"):
            data = json.loads(path.read_text())
            if "script_sha256" not in data:
                continue
            if data["script_sha256"] != sha(ROOT / "archive/paired_benchmark.py"):
                raise ValueError(f"Historical measurement-script checksum differs: {path}")
            if data["reference_script_sha256"] != sha(ROOT / "archive/compare_optimized.py"):
                raise ValueError(f"Historical reference-script checksum differs: {path}")
            records += 1
        if records == 0:
            raise ValueError("No historical paired case JSON records were found")
    print(f"Verified {len(manifest['files'])} frozen files and {records} historical measurement records")


if __name__ == "__main__":
    main()
