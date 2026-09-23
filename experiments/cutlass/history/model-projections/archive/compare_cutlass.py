"""Compare tuned CUTLASS collective, direct CuTe, and Triton INT8 G256 kernels.

Run one shape per process (M=1024/2048 model projections).
JSON checkpoints are durable. All timings use prequantized identical inputs,
FP32 group scales, BF16 output, preallocated output and warm CUDA graph replay.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


BACKENDS = ("cutlass", "cuda", "triton")
SHAPES = [(1024, 1024, 1536), (1024, 1536, 1024), (1024, 2048, 1536), (1024, 3072, 1536), (1024, 512, 1536), (1024, 4096, 1536), (1024, 1536, 3072), (1024, 151936, 1536), (1024, 262144, 1536), (2048, 1024, 1536), (2048, 1536, 1024), (2048, 2048, 1536), (2048, 3072, 1536), (2048, 512, 1536), (2048, 4096, 1536), (2048, 1536, 3072), (2048, 151936, 1536), (2048, 262144, 1536)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def provenance():
    import grain_gemm
    root = Path(grain_gemm.__file__).parent
    return dict(
        script_sha256=digest(Path(__file__)),
        candidate_script_sha256=digest(Path(__file__).with_name('tune_grain.py')),
        sources={str(p.relative_to(root)): digest(p) for p in root.rglob('*')
                 if p.is_file() and p.suffix in ('.py', '.cu', '.hpp', '.json')},
        libraries={name: digest(root / 'kernels/_native' / name)
                   for name in ['libgrain_cuda.so', 'libgrain_cutlass.so']},
    )


def environment(torch, triton):
    try:
        driver = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return dict(device=torch.cuda.get_device_name(),
                compute_capability=list(torch.cuda.get_device_capability()),
                multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
                torch=torch.__version__, triton=triton.__version__,
                cuda=torch.version.cuda, driver=driver)


def reference(a, b, sa, sb, torch):
    rows = torch.linspace(0, a.shape[0]-1, min(32, a.shape[0]), device=a.device).long()
    cols = torch.linspace(0, b.shape[1]-1, min(32, b.shape[1]), device=b.device).long()
    aa, bb = a[rows].cpu().int(), b[:, cols].cpu().int()
    saa, sbb = sa[rows].cpu(), sb[:, cols].cpu()
    expected = torch.zeros((len(rows), len(cols)), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], 256)):
        dot = aa[:, start:start+256] @ bb[start:start+256]
        scale = saa[:, group, None] * sbb[group, None, :]
        expected = (dot.double()*scale.double()+expected.double()).float()
    return rows, cols, expected.to(torch.bfloat16)


def check(out, ref, torch):
    rows, cols, expected = ref
    actual = out[rows][:, cols].cpu()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(out).all().item()
    return dict(checked_rows=len(rows), checked_columns=len(cols),
                max_abs_error=(actual.float()-expected.float()).abs().max().item())


def capture(function, torch):
    # Bound graph duration even at 16384^3; large kernels use a single node.
    for _ in range(3):
        function()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(3):
        function()
    end.record()
    torch.cuda.synchronize()
    estimate_ms = start.elapsed_time(end) / 3
    nodes = max(1, min(256, int(5 / max(estimate_ms, 0.001))))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            function()
    graph.replay()
    torch.cuda.synchronize()
    return graph, nodes


def time_round(graph, nodes, replays, torch):
    graph.replay()
    events = []
    for _ in range(replays):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end)*1000/nodes for start, end in events]


def summarize(rounds, operations):
    medians = [statistics.median(r) for r in rounds]
    value = statistics.median(medians)
    assert math.isfinite(value) and value > 0
    return dict(rounds_us=rounds, round_medians_us=medians, median_us=value,
                min_round_us=min(medians), max_round_us=max(medians),
                effective_tops=operations/(value*1e6))


def run_case(shape, output_dir):
    import torch
    import triton
    from triton.runtime.errors import OutOfResources
    from grain_gemm.kernels import cuda, cutlass
    from grain_gemm.kernels.triton import launch
    from tune_grain import TRITON_CANDIDATES

    assert torch.cuda.get_device_capability() == (12, 1)
    assert cuda.is_available() and cutlass.is_available()
    path = output_dir / ('case_'+'_'.join(map(str, shape))+'.json')
    source = provenance()
    runtime = environment(torch, triton)
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get('complete') and old['source'] == source
                and old.get('environment') == runtime):
            print(f'{shape}: retained complete matching checkpoint', flush=True)
            return
        raise RuntimeError(f'Archive the stale/incomplete checkpoint before retrying: {path}')
    m,n,k = shape
    torch.manual_seed(20260923)
    a = torch.randint(-127, 128, (m,k), dtype=torch.int8, device='cuda')
    b = torch.randint(-127, 128, (n,k), dtype=torch.int8, device='cuda').T
    sa = torch.rand((m,k//256), device='cuda')*.009+.001
    sb = torch.rand((k//256,n), device='cuda')*.009+.001
    out = torch.empty((m,n), dtype=torch.bfloat16, device='cuda')
    ref = reference(a,b,sa,sb,torch)
    candidates = [dict(backend='cutlass',config_id=i) for i in range(len(cutlass.CONFIGS))]
    candidates += [dict(backend='cuda',config_id=i) for i in range(len(cuda.CONFIGS))]
    candidates += TRITON_CANDIDATES
    valid, functions, graphs, skipped, checks = [], [], [], [], []
    operations = 2*m*n*k
    print(f'{shape}: compile/check {len(candidates)} candidates', flush=True)
    for config in candidates:
        backend = config['backend']
        if backend in ('cuda','cutlass'):
            module = cuda if backend=='cuda' else cutlass
            def function(cfg=config, native=module):
                return native.launch(a,b,sa,sb,out,cfg['config_id'])
        else:
            def function(cfg=config):
                return launch(a,b,sa,sb,out,256,cfg)
        try:
            function()
        except OutOfResources as error:
            skipped.append(dict(config=config,reason=str(error)))
            continue
        checks.append(check(out,ref,torch))
        graphs.append(capture(function,torch))
        functions.append(function)
        valid.append(config)
    tune = [[] for _ in valid]
    for round_id in range(3):
        for step in range(len(valid)):
            i=(step+round_id)%len(valid)
            tune[i].append(time_round(*graphs[i],3,torch))
        print(f'  tuning {round_id+1}/3',flush=True)
    tuning = [dict(config=config,correctness=check_,graph_nodes=graph[1],
                   **summarize(times,operations))
              for config,check_,graph,times in zip(valid,checks,graphs,tune)]
    selected = {backend:min((i for i,c in enumerate(valid) if c['backend']==backend),
                            key=lambda i:tuning[i]['median_us']) for backend in BACKENDS}
    # Candidate choice is fixed before changing inputs for the final comparison.
    torch.manual_seed(20260924)
    a.random_(-127,128); b.T.random_(-127,128)
    sa.uniform_(.001,.01); sb.uniform_(.001,.01)
    ref = reference(a,b,sa,sb,torch)
    correctness = {}
    saved_a = a.clone()
    for backend,i in selected.items():
        graph,_ = graphs[i]
        graph.replay()
        correctness[backend] = check(out,ref,torch)
        a.zero_()
        graph.replay()
        assert torch.count_nonzero(out).item()==0
        a.copy_(saved_a)
        graph.replay()
        check(out,ref,torch)
    del saved_a
    final = {backend:[] for backend in BACKENDS}
    for round_id in range(5):
        for offset in range(len(BACKENDS)):
            backend=BACKENDS[(offset+round_id)%len(BACKENDS)]
            final[backend].append(time_round(*graphs[selected[backend]],5,torch))
        print(f'  final {round_id+1}/5',flush=True)
    results = {backend:dict(config=valid[selected[backend]],correctness=correctness[backend],
                           graph_nodes=graphs[selected[backend]][1],
                           **summarize(final[backend],operations)) for backend in BACKENDS}
    report = dict(complete=True,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  shape=dict(m=m,n=n,k=k),group_size=256,output_dtype='bfloat16',
                  device=torch.cuda.get_device_name(),compute_capability=[12,1],
                  software=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda),
                  source=source,environment=runtime,seeds=dict(tuning=20260923,final=20260924),
                  timing=dict(method='CUDA graphs',tuning_rounds=3,tuning_replays=3,
                              final_rounds=5,final_replays=5,max_graph_nodes=256,
                              target_graph_ms=5,reduction='median of round medians',
                              order='rotated each round',warm_cache=True,preallocated_output=True,
                              scope='GEMM, group scaling and BF16 conversion; no quantization or host dispatch'),
                  tuning=tuning,skipped=skipped,results=results)
    save(report,path)
    for backend,r in results.items():
        print(f"  {backend}: {r['median_us']:.3f} us / {r['effective_tops']:.2f} TOPS / {r['config']}",flush=True)


def render(output_dir):
    cases = [json.loads(path.read_text()) for path in output_dir.glob('case_*.json')]
    if not cases:
        raise ValueError(f'No completed case JSON files in {output_dir}')
    for case in cases:
        if not case.get('complete'):
            raise ValueError('Refusing to render an incomplete case')
        for field in ('source', 'software', 'device', 'compute_capability', 'environment', 'timing'):
            if case.get(field) != cases[0].get(field):
                raise ValueError(f'Refusing to mix cases with different {field}')
    cases.sort(key=lambda c:tuple(c['shape'].values()))
    lines = ['# GB10 G256: CUTLASS collective vs direct CuTe vs Triton', '',
             'Pure prequantized INT8 GEMM, FP32 group scales and accumulation, BF16 output. '
             'Each backend is tuned separately on the same input; selected candidates are timed '
             'again on fresh input. CUDA graphs use warm caches and preallocated output. '
             'Compilation, tuning, quantization, allocation and host dispatch are excluded.', '',
             'Latency is μs. Equivalent TFLOPS uses 2MNK/time; the actual INT8 throughput unit is TOPS.', '',
             '| M | N | K | CUTLASS μs | CuTe μs | Triton μs | CUTLASS / CuTe speedup | Equivalent TFLOPS: CUTLASS / CuTe / Triton |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    csv_rows = []
    for case in cases:
        m,n,k=case['shape'].values()
        results=case['results']
        c,u,t=(results[backend] for backend in BACKENDS)
        lines.append(f"| {m} | {n} | {k} | {c['median_us']:.3f} | {u['median_us']:.3f} | {t['median_us']:.3f} | {u['median_us']/c['median_us']:.3f}× | {c['effective_tops']:.2f} / {u['effective_tops']:.2f} / {t['effective_tops']:.2f} |")
        row=dict(m=m,n=n,k=k)
        for backend,r in results.items():
            row.update({f'{backend}_us':r['median_us'],f'{backend}_equivalent_tflops':r['effective_tops'],
                        f'{backend}_config':json.dumps(r['config'],sort_keys=True)})
        csv_rows.append(row)
    lines += ['', 'Speedup = CuTe latency / CUTLASS latency; greater than 1 means CUTLASS is faster.', '',
              'These measurements compare the implementations here, not a universal ranking of frameworks. '
              'CUTLASS uses a custom SM80-style collective and GemmUniversal composition compiled for SM121; '
              'both native backends use INT8 mma.sync and cp.async. The CUTLASS path uses the upstream '
              'vectorized epilogue. It does not invoke the direct CuTe kernel.', '',
              'Correctness: independent CPU INT32 group dots with FP32 FMA rounding at up to 32×32 sampled '
              'outputs, full-output finite checks, and zero/restore CUDA graph replay for all selected kernels. '
              'Full per-round timings, candidates, hashes and build versions are in each case JSON.', '']
    lines += ['## Native timing ranges', '',
              'Ranges are the minimum and maximum of five round medians, not confidence intervals. '
              'Some large shapes vary by several percent between rounds; small performance gaps '
              'should not be generalized to other devices, builds or workloads.', '',
              '| M | N | K | CUTLASS round range μs | CuTe round range μs | CUTLASS config | CuTe config |',
              '|---:|---:|---:|---|---|---:|---:|']
    for case in cases:
        m,n,k=case['shape'].values()
        c,u=(case['results'][b] for b in ('cutlass','cuda'))
        lines.append(f"| {m} | {n} | {k} | {c['min_round_us']:.3f}–{c['max_round_us']:.3f} | {u['min_round_us']:.3f}–{u['max_round_us']:.3f} | {c['config']['config_id']} | {u['config']['config_id']} |")
    (output_dir/'report.md').write_text('\n'.join(lines))
    if csv_rows:
        with (output_dir/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
            writer.writeheader();writer.writerows(csv_rows)
    print(f'Rendered {len(cases)}/{len(SHAPES)} cases at {output_dir}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--case',type=int,nargs=3,metavar=('M','N','K'))
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    if args.render_only:
        render(args.output_dir)
    elif args.case:
        if any(x<=0 for x in args.case) or args.case[0]%128 or args.case[1]%128 or args.case[2]%256:
            parser.error('candidate sweep requires positive M/N multiples128 and K multiple256')
        run_case(tuple(args.case),args.output_dir)
    else:
        for shape in SHAPES:
            subprocess.run([sys.executable,__file__,'--output-dir',str(args.output_dir),
                            '--case',*map(str,shape)],check=True)
            render(args.output_dir)


if __name__=='__main__':
    main()
