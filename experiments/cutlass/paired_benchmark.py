"""Fixed-config, interleaved measurement of CuTe/CUTLASS and isolated changes.

No tuning or package modifications. All variants use exactly the same tensors
and number of CUDA Graph nodes. Repeated inputs, warm caches, BF16 output.
"""

import argparse
import ctypes
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD = ROOT / "build"
SHAPES = [(1024, 1024, 1536, 3, 11), (1024, 1536, 1024, 7, 15),
          (1024, 2048, 1536, 7, 15), (2048, 1536, 1024, 7, 15),
          (2048, 151936, 1536, 15, 23)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def telemetry():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def bootstrap_interval(values, seed=196):
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(4000))
    return [draws[100], draws[3899]]


class Native:
    def __init__(self, path, symbol):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.fn = getattr(self.lib, symbol)
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def launch(self, a, b, sa, sb, out, config, torch):
        code = self.fn(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                       out.data_ptr(), a.shape[0], b.shape[1], a.shape[1], config,
                       torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f'{self.path.name}: native return {code}')


def run(shape, libraries, output, rounds):
    path = output / ('case_%d_%d_%d.json' % tuple(shape[:3]))
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite measurement: {path}')
    import torch
    from reference import check, reference

    m, n, k, cute_id, cutlass_id = shape
    torch.manual_seed(20260926)
    a = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8).T
    sa = torch.rand((m, k // 256), device='cuda') * .009 + .001
    sb = torch.rand((k // 256, n), device='cuda') * .009 + .001
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref = reference(a, b, sa, sb, torch)
    functions = {}
    correctness = {}
    estimates = []
    for name, lib in libraries.items():
        cfg = cute_id if name == 'cute' else cutlass_id
        def function(native=lib, config=cfg):
            native.launch(a, b, sa, sb, out, config, torch)
        functions[name] = function
        function()
        correctness[name] = check(out, ref, torch)
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(5):
            function()
        end.record()
        torch.cuda.synchronize()
        estimates.append(start.elapsed_time(end) / 5)
    nodes = max(1, min(512, int(10 / max(estimates))))
    graphs = {}
    for name, fn in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                fn()
        graph.replay()
        check(out, ref, torch)
        graphs[name] = graph
    names = list(graphs)
    # Bring the device out of idle before comparing sub-percent differences.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    times = {name: [] for name in names}
    orders = []
    sensors = []
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(3)]
    for round_id in range(rounds):
        order = names[round_id % len(names):] + names[:round_id % len(names)]
        if (round_id // len(names)) % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            graph = graphs[name]
            graph.replay()
            for start, end in pairs:
                start.record(); graph.replay(); end.record()
            torch.cuda.synchronize()
            times[name].append([start.elapsed_time(end) * 1000 / nodes for start, end in pairs])
        if round_id % 10 == 0:
            sensors.append(dict(round=round_id, temperature_clock_power_util=telemetry()))
    medians = {name: [statistics.median(r) for r in rows] for name, rows in times.items()}
    results = {}
    for name, values in medians.items():
        ratios = [x / y for x, y in zip(values, medians['cute'])]
        results[name] = dict(config_id=cute_id if name == 'cute' else cutlass_id,
                             rounds_us=times[name], round_medians_us=values,
                             median_us=statistics.median(values), min_us=min(values), max_us=max(values),
                             paired_ratio_to_cute=statistics.median(ratios),
                             bootstrap_95_interval=bootstrap_interval(ratios),
                             correctness=correctness[name])
    contrasts = {}
    base = 'cutlass_matched' if 'cutlass_matched' in medians else 'cutlass_production'
    for name in names:
        ratios = [x / y for x, y in zip(medians[name], medians[base])]
        contrasts[name] = dict(reference=base, median_ratio=statistics.median(ratios),
                               bootstrap_95_interval=bootstrap_interval(ratios))
    report = dict(shape=dict(m=m, n=n, k=k), group_size=256, output_dtype='bfloat16',
                  fixed_configs=dict(cute=cute_id, cutlass=cutlass_id),
                  library_hashes={name: sha(lib.path) for name, lib in libraries.items()},
                  script_sha256=sha(Path(__file__)), reference_script_sha256=sha(ROOT / 'reference.py'),
                  device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  nodes=nodes, rounds=rounds, warmup_seconds=2, orders=orders, sensors=sensors,
                  results=results, contrasts=contrasts,
                  note='Bootstrap intervals resample paired round ratios; adjacent rounds may be correlated. '
                       'Intervals describe this run, not independent machine/session replication. '
                       'Pure prequantized GEMM with warm caches and preallocated output; no host dispatch timed.')
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'case_{m}_{n}_{k}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print((m, n, k), 'nodes', nodes, flush=True)
    for name, row in results.items():
        print(name, round(row['median_us'], 4), 'us',
              round((row['paired_ratio_to_cute'] - 1) * 100, 3), '% vs CuTe', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--case', nargs=5, type=int, metavar=('M', 'N', 'K', 'CUTE_ID', 'CUTLASS_ID'))
    parser.add_argument('--cute-library', type=Path,
                        default=DEFAULT_BUILD / 'cute_control/libgrain_cuda.so')
    parser.add_argument('--cutlass-library', type=Path,
                        default=DEFAULT_BUILD / 'cutlass_control/libgrain_cutlass.so')
    parser.add_argument('--variant', action='append', default=[], metavar='NAME=LIBRARY')
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error('--rounds must be positive')
    variants = {}
    for item in args.variant:
        name, sep, path = item.partition('=')
        if not sep or not name or not path or name in ('cute', 'cutlass_production') or name in variants:
            parser.error('--variant requires a unique NAME=LIBRARY; cute/cutlass_production are reserved')
        variants[name] = Path(path).expanduser().resolve()
    if args.case:
        m, n, k, cute_id, cutlass_id = args.case
        if min(m, n, k) <= 0 or m % 128 or n % 128 or k % 256:
            parser.error('The paired experiment requires positive M/N multiples of 128 and K multiple of 256')
        if (cute_id, cutlass_id) not in ((3, 11), (7, 15), (15, 23)):
            parser.error('Supported matched config pairs: 3/11, 7/15, 15/23')
        libraries = {
            'cute': Native(args.cute_library.expanduser().resolve(), 'grain_cute_g256'),
            'cutlass_production': Native(args.cutlass_library.expanduser().resolve(), 'grain_cutlass_g256'),
        }
        libraries.update({name: Native(path, 'grain_cutlass_g256') for name, path in variants.items()})
        run(args.case, libraries, args.output, args.rounds)
    else:
        for shape in SHAPES:
            command = [sys.executable, __file__, '--output', str(args.output), '--rounds', str(args.rounds),
                       '--cute-library', str(args.cute_library), '--cutlass-library', str(args.cutlass_library),
                       '--case', *map(str, shape)]
            for name, path in variants.items():
                command += ['--variant', f'{name}={path}']
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
