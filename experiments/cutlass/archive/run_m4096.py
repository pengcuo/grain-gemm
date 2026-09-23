"""Extend the fixed-config residual-gap experiment to M=4096, locally."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SHAPES = [(4096, 1024, 1536, 3, 11), (4096, 1536, 1024, 7, 15),
          (4096, 2048, 1536, 7, 15), (4096, 151936, 1536, 15, 23)]
VARIANTS = {
    'cutlass_matched': 'baseline',
    'cutlass_minimum_k': 'minimum_k_guard',
    'cutlass_no_tail': 'no_direct_tail_barrier',
}


def main():
    output = ROOT / 'paired_m4096'
    output.mkdir(exist_ok=True)
    commands = []
    for shape in SHAPES:
        path = output / ('case_%d_%d_%d.json' % shape[:3])
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite measurement: {path}')
        command = [sys.executable, str(ROOT / 'paired_benchmark.py'),
                   '--output', str(output), '--rounds', '60',
                   '--case', *map(str, shape)]
        for name, directory in VARIANTS.items():
            command += ['--variant', f'{name}={ROOT / directory / "libgrain_cutlass.so"}']
        commands.append(command)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '4'
    env['PYTHONPATH'] = '/home/pengcuo/spark/grain-gemm/src'
    for command in commands:
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
