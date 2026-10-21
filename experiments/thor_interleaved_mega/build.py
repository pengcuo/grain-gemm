#!/usr/bin/env python3
"""Build the isolated SM110 interleaved Mega prototype; no GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
CUTLASS_COMMIT='098de2a652cf8f00fd70b2df54051c7eccbb855a'
ABI_VERSION=1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {path.name:sha(path) for path in sorted(HERE.iterdir())
            if path.suffix in ('.cu','.cuh','.hpp','.h')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutlass-dir',type=Path,required=True)
    parser.add_argument('--nvcc',default=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc')
    parser.add_argument('--arch',choices=('sm_110a',),default='sm_110a')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--silu-lut',action='store_true',help='Use the exact BF16-gate SiLU table; separate build/lut output by default')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cutlass=args.cutlass_dir.expanduser().resolve()
    if not (cutlass/'include/cute/tensor.hpp').is_file():
        parser.error('CUTLASS headers were not found')
    revision=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if revision!=CUTLASS_COMMIT:
        parser.error('Require CUTLASS '+CUTLASS_COMMIT+'; got '+revision)
    if subprocess.check_output(['git','-C',str(cutlass),'diff','HEAD','--','include'],text=True):
        parser.error('Require unmodified tracked CUTLASS include files')
    nvcc=shutil.which(args.nvcc) or args.nvcc
    if not Path(nvcc).is_file():
        parser.error('CUDA compiler was not found: '+nvcc)
    output_dir=(args.output_dir or (HERE/'build/lut' if args.silu_lut else HERE/'build')).expanduser().resolve()
    output=output_dir/'libgrain_moe_mega.so'
    flags=[nvcc,'-O3','-std=c++17','--expt-relaxed-constexpr',
           '-gencode=arch=compute_110a,code=sm_110a','-Xcompiler=-fPIC',
           '-shared','-fmad=false',f'-I{cutlass/"include"}','--ptxas-options=-v',
           str(HERE/'mega_cluster.cu'),'-lcuda']
    if args.silu_lut:
        flags.append('-DGRAIN_MOE_EXACT_SILU_LUT')
    if args.dry_run:
        print(json.dumps(dict(command=flags+['-o',str(output)],cutlass_commit=revision,
            source_sha256=source_hashes()),indent=2))
        return 0
    output_dir.mkdir(parents=True,exist_ok=True)
    original_hashes=source_hashes()
    compiler=subprocess.check_output([nvcc,'--version'],text=True).strip()
    with tempfile.TemporaryDirectory(prefix='mega-build-',dir=output_dir) as temporary:
        built=Path(temporary)/output.name
        command=flags+['-o',str(built)]
        subprocess.run(command,check=True)
        after_revision=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
        after_diff=subprocess.check_output(['git','-C',str(cutlass),'diff','HEAD','--','include'],text=True)
        if after_revision!=revision or after_diff:
            raise RuntimeError('CUTLASS revision/headers changed while compiling; rerun the build')
        if source_hashes()!=original_hashes:
            raise RuntimeError('Experiment sources changed while compiling')
        metadata=dict(architecture='sm_110',compiler_architecture='sm_110a',
            abi_version=ABI_VERSION,consumer_warps=[8,16],group_size=256,
            gate_up_layout='interleaved',hidden=1536,intermediate=1024,silu_lut=args.silu_lut,
            cutlass_commit=revision,nvcc_version=compiler,compile_command=command,
            source_sha256=original_hashes,library_sha256=sha(built),
            experiment='Cluster-local G1/SwiGLU/INT8-G256/G2; separate combine; explicit experiment only')
        metadata_file=Path(temporary)/'libgrain_moe_mega.json'
        metadata_file.write_text(json.dumps(metadata,indent=2)+'\n')
        built.replace(output)
        metadata_file.replace(output.with_suffix('.json'))
    print('Built '+str(output))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
