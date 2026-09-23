#!/usr/bin/env python3
"""Build isolated CUTLASS variants and the frozen CuTe/CUTLASS controls."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
CUTLASS_COMMIT = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
VARIANTS = (
    "baseline", "no_direct_tail_barrier", "bf16_uint4_i32_address",
    "bf16_fixed_store_loop", "late_fp32_accumulator_clear",
    "minimum_k_guard", "ungrouped_specialization",
)
TARGETS = ("cute_control", "cutlass_control", *VARIANTS)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-dir", type=Path, required=True,
                        help=f"CUTLASS checkout; measured revision: {CUTLASS_COMMIT}")
    parser.add_argument("--arch", default="sm_121")
    parser.add_argument("--nvcc")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--target", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without compiling or writing")
    args = parser.parse_args()
    if not re.fullmatch(r"sm_[0-9]+[af]?", args.arch):
        parser.error("--arch must name a CUDA architecture, e.g. sm_121")
    cutlass = args.cutlass_dir.expanduser().resolve()
    if not (cutlass / "include/cutlass/gemm/device/gemm_universal_adapter.h").is_file():
        parser.error(f"CUTLASS headers were not found under {cutlass / 'include'}")
    nvcc = args.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not args.dry_run and not Path(nvcc).is_file() and shutil.which(nvcc) is None:
        parser.error(f"CUDA compiler was not found: {nvcc}")
    revision = None
    if shutil.which("git"):
        git = subprocess.run(["git", "-C", str(cutlass), "rev-parse", "HEAD"],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if git.returncode == 0:
            revision = git.stdout.strip()
    version = None if args.dry_run else subprocess.check_output([nvcc, "--version"], text=True).strip()
    for target in args.target:
        is_cute = target == "cute_control"
        source_dir = (ROOT / "measured_controls" / ("cute" if is_cute else "cutlass")
                      if target.endswith("_control") else ROOT / "variants" / target)
        source = source_dir / ("grain_cute.cu" if is_cute else "grain_cutlass.cu")
        library_name = "libgrain_cuda.so" if is_cute else "libgrain_cutlass.so"
        output_dir = args.build_dir.expanduser().resolve() / target
        output = output_dir / library_name
        # Match tools/build_cuda.py and tools/build_cutlass.py compiler options.
        command = [nvcc, "-O3", "-std=c++17"]
        if not is_cute:
            command.append("--expt-relaxed-constexpr")
        command += [f"-arch={args.arch}", "-Xcompiler=-fPIC", "-shared",
                    f"-I{cutlass / 'include'}", "--ptxas-options=-v", str(source), "-o"]
        print(shlex.join([*command, str(output)]), flush=True)
        if args.dry_run:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compile-", dir=output_dir) as temporary:
            built = Path(temporary) / library_name
            subprocess.run([*command, str(built)], check=True)
            built.replace(output)
        metadata = dict(
            target=target, architecture=args.arch, cutlass_commit=revision,
            tested_cutlass_commit=CUTLASS_COMMIT, nvcc_version=version,
            command=[*command, str(output)], library_sha256=sha(output),
            source_sha256={path.name: sha(path) for path in sorted(source_dir.iterdir())
                           if path.suffix in (".cu", ".hpp")},
        )
        (output_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
