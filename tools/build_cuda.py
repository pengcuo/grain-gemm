#!/usr/bin/env python3
"""Explicitly build GrainGEMM's optional CUDA library against local CUTLASS headers."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


CUTLASS_COMMIT = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-dir", type=Path, required=True,
                        help=f"Local CUTLASS checkout; tested commit: {CUTLASS_COMMIT}")
    parser.add_argument("--arch", default="sm_121", help="Only sm_121 is supported (GB10-only native implementation)")
    parser.add_argument("--nvcc", help="CUDA compiler path; otherwise resolve nvcc or /usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    if args.arch != "sm_121":
        parser.error("This is a GB10-only native implementation; --arch must be sm_121. "
                     "Other architectures require a separate implementation.")
    cutlass = args.cutlass_dir.expanduser().resolve()
    if not (cutlass / "include" / "cute" / "tensor.hpp").is_file():
        parser.error(f"CUTLASS CuTe headers were not found under {cutlass / 'include'}")
    nvcc = args.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file() and shutil.which(nvcc) is None:
        parser.error(f"CUDA compiler was not found: {nvcc}")
    source = ROOT / "src/grain_gemm/kernels/csrc/grain_cute.cu"
    output_dir = ROOT / "src/grain_gemm/kernels/_native"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "libgrain_cuda.so"
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    revision = None
    if shutil.which("git"):
        result = subprocess.run(["git", "-C", str(cutlass), "rev-parse", "HEAD"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            revision = result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="build-", dir=output_dir) as temporary:
        built = Path(temporary) / output.name
        command = [nvcc, "-O3", "-std=c++17", f"-arch={args.arch}",
                   "-Xcompiler=-fPIC", "-shared", f"-I{cutlass / 'include'}",
                   "--ptxas-options=-v", str(source), "-o", str(built)]
        print(f"Building for {args.arch} with CUTLASS {revision or 'revision unknown'}", flush=True)
        subprocess.run(command, check=True)
        built.replace(output)
    metadata = {
        "architecture": args.arch,
        "cutlass_commit": revision,
        "tested_cutlass_commit": CUTLASS_COMMIT,
        "nvcc_version": version,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {output}")


if __name__ == "__main__":
    main()
