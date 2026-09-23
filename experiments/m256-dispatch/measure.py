"""Local fixed-input CuTe grid-granularity experiment; no source edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT / "benchmarks"))
from _device import require_gb10
sys.path.insert(0, str(ROOT.parent / 'prefill'))
from benchmark import reference, validate
import torch
import triton
import grain_gemm
from grain_gemm.kernels import cuda
from grain_gemm.kernels.triton import launch as triton_launch

def snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,pstate',
                                    '--format=csv,noheader'], text=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=256)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, default=1536)
    parser.add_argument('--suite', choices=['main', 'n_sweep', 'm_sweep'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--occupancy', type=Path, default=PROJECT / 'benchmarks/results/m256-dispatch/occupancy.json')
    args = parser.parse_args()
    if args.output_dir.resolve() == (PROJECT / 'benchmarks/results').resolve() or (PROJECT / 'benchmarks/results').resolve() in args.output_dir.resolve().parents:
        parser.error('Write new measurements outside the archived benchmarks/results directory')
    if not (0 < args.m <= 1024 and args.m % 128 == 0 and 0 < args.n <= 4096 and args.n % 128 == 0 and args.k > 0 and args.k % 256 == 0):
        parser.error('This fixed-prefix diagnostic requires M<=1024 and N<=4096, M/N multiples of 128, positive K multiple of 256')
    require_gb10(torch)
    if not cuda.is_available():
        parser.error('Build the native CuTe library first')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupancy = json.loads(args.occupancy.read_text())
    if occupancy['native_so_sha256'] != hashlib.sha256(cuda._LIBRARY.read_bytes()).hexdigest():
        parser.error('Occupancy record must match the active native binary SHA256; query fresh resources for a rebuilt library')
    if occupancy['device']['sm_count'] != torch.cuda.get_device_properties(0).multi_processor_count:
        parser.error('Occupancy device SM count does not match the active device')
    m, n, k = args.m, args.n, args.k
    target = args.output_dir / f'{args.suite}_{m}_{n}_{k}.json'
    if target.exists():
        raise RuntimeError(f'Refusing to replace existing results: {target}')
    torch.manual_seed(20260923)
    # Each shape uses the same deterministic prefix of these maximum tensors.
    a = torch.randint(-127, 128, (1024, k), device='cuda', dtype=torch.int8)[:m]
    b = torch.randint(-127, 128, (4096, k), device='cuda', dtype=torch.int8)[:n].T
    sa = (0.001 + torch.rand((1024, k // 256), device='cuda') * 0.009)[:m].contiguous()
    sb = (0.001 + torch.rand((k // 256, 4096), device='cuda') * 0.009)[:, :n].contiguous()
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref = reference(a, b, sa, sb, torch)
    ids = list(range(8)) + [15] if args.suite == 'main' else [3, 5, 7]
    variants = [(f'cute_{i}', dict(backend='cuda', config_id=i)) for i in ids]
    if args.suite == 'main':
        for name, bm, bn, w in [('triton_default', 64, 128, 8),
                                ('triton_64x128_w4', 64, 128, 4),
                                ('triton_128x64_w4', 128, 64, 4)]:
            variants.append((name, dict(backend='triton', block_m=bm, block_n=bn,
                                       num_warps=w, num_stages=3, swizzle=8)))
    records = {}
    graphs = {}
    nodes, rounds, replays = 256, 5, 9
    defaults = {backend: grain_gemm.get_kernel_config(a, b, group_size=256, output_dtype=torch.bfloat16,
                scale_a=sa, scale_b=sb, backend=backend) for backend in ['auto', 'triton', 'cuda']}
    print(f'{args.suite}: M,N,K={m},{n},{k}; compile/check {len(variants)} variants', flush=True)
    for name, config in variants:
        if config['backend'] == 'cuda':
            def fn():
                return cuda.launch(a, b, sa, sb, out, config['config_id'])
            cfg = cuda.CONFIGS[config['config_id']]
        else:
            def fn():
                return triton_launch(a, b, sa, sb, out, 256, config)
            cfg = config
        compiled = fn()
        check = validate(out, ref, torch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                fn()
        saved_a = a.clone()
        a.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out).item() == 0
        a.copy_(saved_a)
        graph.replay()
        validate(out, ref, torch)
        graphs[name] = graph
        resources = None
        if config['backend'] == 'cuda':
            resources = next(r for r in occupancy['configs'] if r['config_id'] == config['config_id'] and not r['grouped'])
        else:
            resources = dict(registers_per_thread=compiled.n_regs, shared_bytes=compiled.metadata.shared,
                             num_warps=compiled.metadata.num_warps)
        records[name] = dict(config=config, tile=[cfg['block_m'], cfg['block_n']],
                             ctas=(m // cfg['block_m']) * (n // cfg['block_n']),
                             resources=resources, correctness=check, rounds_us=[])
    for _ in range(3):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    environment_before = snapshot()
    names = list(graphs)
    for round_index in range(rounds):
        for offset in range(len(names)):
            name = names[(offset + round_index) % len(names)]
            # Warm the exact variant immediately before the measured replays.
            graphs[name].replay()
            events = []
            for _ in range(replays):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            records[name]['rounds_us'].append([start.elapsed_time(end) * 1000 / nodes for start, end in events])
        print(f'  completed timing round {round_index + 1}/{rounds}', flush=True)
    for name, record in records.items():
        record['round_medians_us'] = [statistics.median(values) for values in record['rounds_us']]
        record['median_us'] = statistics.median(record['round_medians_us'])
        record['min_round_us'] = min(record['round_medians_us'])
        record['max_round_us'] = max(record['round_medians_us'])
        record['equivalent_tflops'] = 2*m*n*k/(record['median_us'] * 1e6)
        print(f"  {name}: {record['median_us']:.4f} us, {record['ctas']} CTAs", flush=True)
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), suite=args.suite,
                  shape=dict(m=m, n=n, k=k), group_size=256, seed=20260923,
                  device=torch.cuda.get_device_name(), sm_count=occupancy['device']['sm_count'],
                  torch_version=torch.__version__, triton_version=triton.__version__,
                  environment_before=environment_before, environment_after=snapshot(),
                  native_sha256=hashlib.sha256(cuda._LIBRARY.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256((PROJECT / 'src/grain_gemm/kernels/csrc/sm12x/int8_g256_cute.cu').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device_check_sha256=hashlib.sha256(Path(require_gb10.__code__.co_filename).read_bytes()).hexdigest(),
                  protocol=dict(graph_nodes=nodes, rounds=rounds, graph_replays_per_round=replays,
                                reduction='median of round medians', variant_order='rotate each round',
                                warm_cache=True, preallocated_output=True, pure_gemm=True),
                  defaults=defaults, variants=records)
    with target.open('w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

if __name__ == '__main__':
    main()
