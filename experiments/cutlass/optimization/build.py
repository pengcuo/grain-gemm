"""Build the three archived native libraries used by the optimization comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutlass-dir', required=True, type=Path)
    parser.add_argument('--nvcc', default=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc')
    parser.add_argument('--arch', default='sm_121', help='Only sm_121 is supported (GB10-only native implementation)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.arch != 'sm_121':
        parser.error('This is a GB10-only native implementation; --arch must be sm_121. '
                     'Other architectures require a separate implementation.')
    include = args.cutlass_dir.expanduser().resolve() / 'include'
    if not (include / 'cute/tensor.hpp').is_file():
        parser.error('CUTLASS headers not found')
    jobs = [
        (ROOT / 'before/src/grain_gemm', 'grain_cutlass.cu', 'libgrain_cutlass.so'),
        (ROOT / 'measurement_sources/grain_gemm', 'grain_cutlass.cu', 'libgrain_cutlass.so'),
        (ROOT / 'measurement_sources/grain_gemm', 'grain_cute.cu', 'libgrain_cuda.so'),
    ]
    metadata = []
    for package, source_name, library in jobs:
        source = package / 'kernels/csrc' / source_name
        target = package / 'kernels/_native' / library
        command = [args.nvcc, '-O3', '-std=c++17', f'-arch={args.arch}',
                   '-Xcompiler=-fPIC', '-shared', f'-I{include}', '--ptxas-options=-v']
        if source_name == 'grain_cutlass.cu':
            command.append('--expt-relaxed-constexpr')
        command.extend([str(source), '-o', str(target)])
        print(json.dumps(command), flush=True)
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            built = Path(temporary) / library
            subprocess.run(command[:-1] + [str(built)], check=True)
            built.replace(target)
        metadata.append({'library': str(target.relative_to(ROOT)),
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'command': command})
    if not args.dry_run:
        (ROOT / 'rebuild_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
