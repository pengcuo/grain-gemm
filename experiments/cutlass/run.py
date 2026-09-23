#!/usr/bin/env python3
"""Reproduce a fixed-config paired suite, writing fresh per-shape measurements."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from paired_benchmark import SHAPES, require_gb10

ROOT = Path(__file__).resolve().parent
M4096 = [(4096, 1024, 1536, 3, 11), (4096, 1536, 1024, 7, 15),
         (4096, 2048, 1536, 7, 15), (4096, 151936, 1536, 15, 23)]
VARIANTS = {
    "cutlass_matched": "baseline",
    "cutlass_no_tail": "no_direct_tail_barrier",
    "cutlass_address32": "bf16_uint4_i32_address",
    "cutlass_fixed_loop": "bf16_fixed_store_loop",
    "cutlass_minimum_k": "minimum_k_guard",
    "cutlass_ungrouped": "ungrouped_specialization",
}
# Preserve the original ordering: rotating/reversing uses this order.
SUITES = {
    "run1": ("cutlass_matched", "cutlass_no_tail", "cutlass_address32", "cutlass_fixed_loop"),
    "run2": ("cutlass_matched", "cutlass_no_tail", "cutlass_fixed_loop", "cutlass_minimum_k", "cutlass_ungrouped"),
    "run3": ("cutlass_matched", "cutlass_no_tail", "cutlass_minimum_k", "cutlass_ungrouped"),
    "m4096": ("cutlass_matched", "cutlass_minimum_k", "cutlass_no_tail"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="m4096")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading CUDA or writing")
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    output = args.output.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    commands = []
    for shape in M4096 if args.suite == "m4096" else SHAPES:
        path = output / ("case_%d_%d_%d.json" % shape[:3])
        if path.exists():
            parser.error(f"Refusing to overwrite measurement: {path}")
        command = [sys.executable, str(ROOT / "paired_benchmark.py"),
                   "--output", str(output), "--rounds", str(args.rounds),
                   "--cute-library", str(build / "cute_control/libgrain_cuda.so"),
                   "--cutlass-library", str(build / "cutlass_control/libgrain_cutlass.so"),
                   "--case", *map(str, shape)]
        for name in SUITES[args.suite]:
            command += ["--variant", f"{name}={build / VARIANTS[name] / 'libgrain_cutlass.so'}"]
        commands.append(command)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    import torch
    require_gb10(torch)
    output.mkdir(parents=True, exist_ok=True)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    for command in commands:
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
