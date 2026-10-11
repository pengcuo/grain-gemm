#!/usr/bin/env python3
"""Explicitly build GrainGEMM's optional CUDA library against local CUTLASS headers."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/grain_gemm/kernels"))
from _native import cuda_sources, cuda_source_hash


CUTLASS_COMMIT = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-dir", type=Path, required=True,
                        help=f"Local CUTLASS checkout; tested commit: {CUTLASS_COMMIT}")
    parser.add_argument("--arch", default="sm_121",
                        help="CUDA target: sm_110 (Thor), sm_121 (GB10, default) or sm_120 (experimental)")
    parser.add_argument("--nvcc", help="CUDA compiler path; otherwise resolve nvcc or /usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    if args.arch not in ("sm_110", "sm_120", "sm_121"):
        parser.error("--arch must be sm_110 (Thor), sm_121 (GB10) or sm_120 (experimental). "
                     "Other architectures require separate support and validation.")
    if args.arch == "sm_120":
        print("Experimental SM120 build: RTX 50-series correctness and performance "
              "have not been validated. Select backend='cuda' explicitly; "
              "auto continues to use Triton on SM120.", flush=True)
    cutlass = args.cutlass_dir.expanduser().resolve()
    if not (cutlass / "include" / "cute" / "tensor.hpp").is_file():
        parser.error(f"CUTLASS CuTe headers were not found under {cutlass / 'include'}")
    nvcc = args.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file() and shutil.which(nvcc) is None:
        parser.error(f"CUDA compiler was not found: {nvcc}")
    source = cuda_sources(args.arch)[0]
    compiler_arch = "sm_110a" if args.arch == "sm_110" else args.arch
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
        command = [nvcc, "-O3", "-std=c++17", f"-arch={compiler_arch}",
                   "-Xcompiler=-fPIC", "-shared", f"-I{cutlass / 'include'}",
                   "--ptxas-options=-v", str(source), "-o", str(built)]
        if args.arch == "sm_110":
            command += ["-DNDEBUG", "--expt-relaxed-constexpr"]
            # Link against a toolkit driver stub when building without a GPU.
            compiler = Path(shutil.which(nvcc) or nvcc).resolve()
            stubs = compiler.parent.parent / "lib64/stubs"
            if stubs.is_dir():
                command.append(f"-L{stubs}")
            command.append("-lcuda")
        print(f"Building for {args.arch} with CUTLASS {revision or 'revision unknown'}", flush=True)
        subprocess.run(command, check=True)
        built.replace(output)
    metadata = {
        "architecture": args.arch,
        "cutlass_commit": revision,
        "tested_cutlass_commit": CUTLASS_COMMIT,
        "nvcc_version": version,
        "source_sha256": cuda_source_hash(args.arch),
        "compiler_architecture": compiler_arch,
    }
    if args.arch == "sm_110":
        metadata["compile_definitions"] = ["NDEBUG"]
    (output_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {output}")


if __name__ == "__main__":
    main()
