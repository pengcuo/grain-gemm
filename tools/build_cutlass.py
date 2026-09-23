#!/usr/bin/env python3
"""Build the optional CUTLASS GemmUniversal backend without replacing CuTe."""

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
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    parser.add_argument("--arch", default="sm_121",
                        help="CUDA target: sm_121 (GB10, default) or sm_120 (experimental, unvalidated)")
    parser.add_argument("--nvcc")
    args = parser.parse_args()
    if args.arch not in ("sm_120", "sm_121"):
        parser.error("--arch must be sm_121 (GB10) or sm_120 (experimental). "
                     "Other architectures require separate support and validation.")
    if args.arch == "sm_120":
        print("Experimental SM120 build: RTX 50-series correctness and performance "
              "have not been validated. Select backend='cutlass' explicitly; "
              "auto continues to use Triton on SM120.", flush=True)
    cutlass = args.cutlass_dir.expanduser().resolve()
    if not (cutlass / "include/cutlass/gemm/device/gemm_universal_adapter.h").is_file():
        parser.error(f"CUTLASS headers were not found under {cutlass / 'include'}")
    nvcc = args.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file() and shutil.which(nvcc) is None:
        parser.error(f"CUDA compiler was not found: {nvcc}")
    source_dir = ROOT / "src/grain_gemm/kernels/csrc/sm12x/cutlass"
    output_dir = ROOT / "src/grain_gemm/kernels/_native"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "libgrain_cutlass.so"
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    revision = None
    if shutil.which("git"):
        result = subprocess.run(["git", "-C", str(cutlass), "rev-parse", "HEAD"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            revision = result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="cutlass-build-", dir=output_dir) as temporary:
        built = Path(temporary) / output.name
        command = [nvcc, "-O3", "-std=c++17", "--expt-relaxed-constexpr", f"-arch={args.arch}",
                   "-Xcompiler=-fPIC", "-shared", f"-I{cutlass / 'include'}",
                   "--ptxas-options=-v",
                   str(source_dir / "int8_g256.cu"), "-o", str(built)]
        print(f"Building CUTLASS backend for {args.arch} with {revision or 'unknown revision'}", flush=True)
        subprocess.run(command, check=True)
        built.replace(output)
    metadata = dict(
        architecture=args.arch, cutlass_commit=revision, tested_cutlass_commit=CUTLASS_COMMIT,
        nvcc_version=version,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(source_dir.iterdir())
                       if path.suffix in (".cu", ".hpp")},
    )
    (output_dir / "cutlass_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {output}")


if __name__ == "__main__":
    main()
