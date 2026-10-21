#!/usr/bin/env python3
"""Build the optional SM110 fused INT8 MoE gate/up and activation library."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CUTLASS_COMMIT = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/grain_gemm/kernels"))
from moe_fused import CONSUMER_WARPS, _ABI_VERSION, _METADATA_NAME, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "src/grain_gemm/kernels/_native")
    args = parser.parse_args()
    cutlass = args.cutlass_dir.expanduser().resolve()
    if not (cutlass / "include/cute/tensor.hpp").is_file():
        parser.error("CUTLASS headers were not found")
    nvcc = args.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file() and shutil.which(nvcc) is None:
        parser.error(f"CUDA compiler was not found: {nvcc}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "libgrain_moe_fused.so"
    source = ROOT / "src/grain_gemm/kernels/csrc/sm110/int8_g256_moe_fused.cu"
    hashes = source_hashes()
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    revision = None
    if shutil.which("git"):
        result = subprocess.run(["git", "-C", str(cutlass), "rev-parse", "HEAD"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            revision = result.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="moe-fused-build-", dir=output_dir) as temporary:
        built = Path(temporary) / output.name
        # Explicit fma operations in the G256 accumulation are retained. Other
        # expressions follow the reference's separate rounding operations.
        command = [nvcc, "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                   "-gencode=arch=compute_110a,code=sm_110a", "-Xcompiler=-fPIC",
                   "-shared", "-fmad=false", f"-I{cutlass/'include'}",
                   "--ptxas-options=-v", str(source), "-lcuda", "-o", str(built)]
        print(f"Building optional Thor fused MoE with consumer warps {CONSUMER_WARPS}", flush=True)
        subprocess.run(command, check=True)
        if source_hashes() != hashes:
            raise RuntimeError("Fused MoE sources changed during compilation; rerun the build")
        metadata = dict(architecture="sm_110", compiler_architecture="sm_110a",
                        abi_version=_ABI_VERSION, consumer_warps=list(CONSUMER_WARPS),
                        cutlass_commit=revision, tested_cutlass_commit=CUTLASS_COMMIT,
                        nvcc_version=version, compile_command=command,
                        source_sha256=hashes,
                        library_sha256=hashlib.sha256(built.read_bytes()).hexdigest())
        metadata_path = Path(temporary) / _METADATA_NAME
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        built.replace(output)
        metadata_path.replace(output.with_name(_METADATA_NAME))
    print(f"Built {output}")


if __name__ == "__main__":
    main()
