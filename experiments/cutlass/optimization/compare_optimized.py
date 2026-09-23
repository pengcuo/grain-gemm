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


# Use the archived package so current public dispatch/default changes cannot alter the experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent / "measurement_sources"))

BACKENDS = ("cutlass_before", "cutlass", "cuda", "triton",
            "cutlass_fp32_shared", "cutlass_bf16_shared",
            "cutlass_async_direct", "cutlass_reg_direct")
BASELINE_LIBRARY = Path(__file__).parent / "before/src/grain_gemm/kernels/_native/libgrain_cutlass.so"
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
        baseline_library_sha256=digest(BASELINE_LIBRARY),
        baseline_manifest_sha256=digest(Path(__file__).parent / "before_hashes.json"),
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
    import ctypes
    baseline = ctypes.CDLL(str(BASELINE_LIBRARY))
    baseline.grain_cutlass_g256.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    baseline.grain_cutlass_g256.restype = ctypes.c_int
    class Baseline:
        @staticmethod
        def launch(a,b,sa,sb,out,config_id):
            status = baseline.grain_cutlass_g256(
                a.data_ptr(),b.data_ptr(),sa.data_ptr(),sb.data_ptr(),out.data_ptr(),
                a.shape[0],b.shape[1],a.shape[1],config_id,torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError(f'Original CUTLASS library returned {status}')
            return out
    m,n,k = shape
    torch.manual_seed(20260923)
    a = torch.randint(-127, 128, (m,k), dtype=torch.int8, device='cuda')
    b = torch.randint(-127, 128, (n,k), dtype=torch.int8, device='cuda').T
    sa = torch.rand((m,k//256), device='cuda')*.009+.001
    sb = torch.rand((k//256,n), device='cuda')*.009+.001
    out = torch.empty((m,n), dtype=torch.bfloat16, device='cuda')
    ref = reference(a,b,sa,sb,torch)
    candidates = [dict(backend='cutlass_before',config_id=i) for i in range(8)]
    candidates += [dict(backend='cutlass',config_id=i) for i in range(len(cutlass.CONFIGS))]
    candidates += [dict(backend='cuda',config_id=i) for i in range(len(cuda.CONFIGS))]
    candidates += TRITON_CANDIDATES
    valid, functions, graphs, skipped, checks = [], [], [], [], []
    operations = 2*m*n*k
    print(f'{shape}: compile/check {len(candidates)} candidates', flush=True)
    for config in candidates:
        backend = config['backend']
        if backend in ('cuda','cutlass','cutlass_before'):
            module = {'cuda':cuda,'cutlass':cutlass,'cutlass_before':Baseline}[backend]
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
                            key=lambda i:tuning[i]['median_us'])
                for backend in ('cutlass_before','cutlass','cuda','triton')}
    for family,begin in [('cutlass_fp32_shared',0),('cutlass_bf16_shared',8),
                         ('cutlass_async_direct',16),('cutlass_reg_direct',24)]:
        selected[family] = min((i for i,c in enumerate(valid)
                               if c['backend']=='cutlass' and begin <= c['config_id'] < begin+8),
                              key=lambda i:tuning[i]['median_us'])
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
    cases = [json.loads(p.read_text()) for p in output_dir.glob('case_*.json')]
    if not cases:
        return
    for c in cases:
        assert c['complete']
        for field in ('source','environment','software','timing'):
            assert c[field] == cases[0][field], field
    cases.sort(key=lambda c:tuple(c['shape'].values()))
    lines = ['# CUTLASS G256 optimization on GB10', '',
             'All backends use identical prequantized inputs, FP32 scales/accumulation, and BF16 output. '
             'The original CUTLASS binary is loaded separately and remeasured in the same run. '
             'Each backend/family is tuned independently, then measured on fresh data in five rotated rounds. '
             'CUDA Graph timings include scaling and output conversion, exclude quantization/compilation/tuning/host dispatch. '
             'Warm caches, preallocated output. Latency is microseconds; equivalent TFLOPS=2MNK/time, numerically INT8 TOPS.', '',
             '| M | N | K | Original CUTLASS us | Optimized CUTLASS us | CuTe us | Triton us | CUTLASS latency reduction | Equivalent TFLOPS: old / new / CuTe / Triton |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    csv_rows=[]
    for c in cases:
        m,n,k=c['shape'].values();r=c['results']
        old,new,cu,tr=[r[b] for b in BACKENDS[:4]]
        times=' | '.join(f"{v['median_us']:.3f}" for v in (old,new,cu,tr))
        tops=' / '.join(f"{v['effective_tops']:.2f}" for v in (old,new,cu,tr))
        lines.append(f"| {m} | {n} | {k} | {times} | {(1-new['median_us']/old['median_us'])*100:.2f}% | {tops} |")
        row=dict(m=m,n=n,k=k)
        for name,v in r.items():
            row.update({f'{name}_us':v['median_us'],f'{name}_equivalent_tflops':v['effective_tops'],
                        f'{name}_config':json.dumps(v['config'],sort_keys=True),
                        f'{name}_min_round_us':v['min_round_us'],f'{name}_max_round_us':v['max_round_us']})
        csv_rows.append(row)
    lines += ['', '## CUTLASS family comparison', '',
              'Each family independently selects its best tile; these are not necessarily same-tile ablations. '
              'Exact same-tile comparisons are available among all 32 candidates in each case JSON.', '',
              '| M | N | K | FP32 shared us (ID) | BF16 shared us (ID) | Async scale + direct BF16 us (ID) | Register scale + direct BF16 us (ID) |',
              '|---:|---:|---:|---|---|---|---|']
    for c in cases:
        m,n,k=c['shape'].values()
        cells=' | '.join(f"{c['results'][b]['median_us']:.3f} ({c['results'][b]['config']['config_id']})" for b in BACKENDS[4:])
        lines.append(f'| {m} | {n} | {k} | {cells} |')
    lines += ['', '## Five-round ranges', '',
              'These are min/max round medians, not confidence intervals.', '',
              '| M | N | K | Original CUTLASS us | Optimized CUTLASS us | CuTe us | Triton us |',
              '|---:|---:|---:|---|---|---|---|']
    for c in cases:
        m,n,k=c['shape'].values()
        cells=' | '.join(f"{c['results'][b]['min_round_us']:.3f}–{c['results'][b]['max_round_us']:.3f}" for b in BACKENDS[:4])
        lines.append(f'| {m} | {n} | {k} | {cells} |')
    lines += ['', 'Correctness: every candidate is checked at 32x32 sampled outputs against an independent CPU '
              'INT32 group reference with FP32 FMA rounding; full output is checked for finite values. '
              'Selected graphs are checked with fresh inputs and zero/restore replay. '
              'Raw JSON retains all timing rounds, source/library hashes, and chosen configurations.', '']
    (output_dir/'report.md').write_text('\n'.join(lines))
    with (output_dir/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(f'Rendered {len(cases)}/{len(SHAPES)} at {output_dir}',flush=True)


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
