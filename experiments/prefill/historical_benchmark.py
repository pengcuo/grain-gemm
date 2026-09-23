"""Local-only, shape-derived LLM prefill G256 benchmark on GB10.

No model weights, routing trace, kernel edits, or dispatch-table updates are used.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


ROOT = Path('/home/pengcuo/spark/grain-gemm')
METHODS = ('triton_default', 'cute_default', 'triton_tuned', 'cute_tuned')
TOKENS = [512, 1024, 2048, 4096]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def workloads(tokens):
    cases = {}

    def add(m, n, k, key, label, category, t):
        case = cases.setdefault((m, n, k), dict(shape=dict(m=m, n=n, k=k), uses=[]))
        case['uses'].append(dict(id=key, label=label, category=category, tokens=t))

    for t in tokens:
        for key, label, n, k in (
            ('q', 'Q projection', 3072, 1536),
            ('k_or_v', 'K or V projection', 512, 1536),
            ('qkv', 'Fused QKV', 4096, 1536),
            ('attention_out', 'Attention output', 1536, 3072),
            ('shared_gate_or_up', 'Shared gate or up', 1024, 1536),
            ('shared_gate_up', 'Shared fused gate + up', 2048, 1536),
            ('shared_down', 'Shared down', 1536, 1024),
            ('router', 'Router projection', 32, 1536),
        ):
            add(t, n, k, key, label, 'dense', t)
        for key, label, n, k in (
            ('expert_gate_or_up', 'One expert gate or up', 1024, 1536),
            ('expert_gate_up', 'One expert fused gate + up', 2048, 1536),
            ('expert_down', 'One expert down', 1536, 1024),
        ):
            add(t // 8, n, k, key, label, 'expert', t)
        for vocab in (151936, 262144):
            add(t, vocab, 1536, f'lm_head_{vocab}', f'All-token LM head ({vocab})', 'head_all', t)
    for vocab in (151936, 262144):
        add(1, vocab, 1536, f'lm_head_last_{vocab}', f'Last-token LM head ({vocab})', 'head_last', 1)
    return list(cases.values())


def triton_candidates():
    return [dict(backend='triton', block_m=m, block_n=n, num_warps=w,
                 num_stages=s, swizzle=8)
            for m, n, w, s in (
                (16, 32, 4, 2), (16, 64, 4, 2), (32, 64, 4, 2),
                (32, 128, 4, 3), (64, 64, 4, 2), (64, 128, 8, 3),
                (64, 128, 4, 3), (128, 64, 4, 3), (128, 128, 8, 2),
            )]


def canonical(config):
    ignored = {'policy', 'compute_capability'}
    return {key: value for key, value in config.items() if key not in ignored}


def summary(samples, operations):
    median = statistics.median(samples)
    return dict(round_medians_ms=samples, median_ms=median, min_ms=min(samples),
                max_ms=max(samples), effective_tops=operations / (median * 1e9))


def save(report, output):
    report['updated_at_utc'] = stamp()
    temporary = output.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    csv_path = output.with_suffix('.csv')
    temporary_csv = csv_path.with_suffix('.csv.tmp')
    with temporary_csv.open('w', newline='') as stream:
        writer = csv.writer(stream)
        fields = ['component', 'category', 'tokens', 'm', 'n', 'k']
        for method in METHODS:
            fields.extend([f'{method}_us', f'{method}_tops'])
        writer.writerow(fields + ['default_speedup', 'tuned_speedup'])
        for case in report['cases']:
            for use in case['uses']:
                row = [use['id'], use['category'], use['tokens'], *case['shape'].values()]
                for method in METHODS:
                    result = case['results'].get(method)
                    row += ([1000 * result['median_ms'], result['effective_tops']]
                            if result else ['', ''])
                writer.writerow(row + [case['default_speedup'], case['tuned_speedup']])
        stream.flush()
        os.fsync(stream.fileno())
    temporary_csv.replace(csv_path)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def reference(a, b, sa, sb, torch):
    rows = torch.linspace(0, a.shape[0] - 1, min(32, a.shape[0]), device=a.device).long()
    cols = torch.linspace(0, b.shape[1] - 1, min(32, b.shape[1]), device=b.device).long()
    aa = a.index_select(0, rows).cpu().int()
    bb = b.index_select(1, cols).cpu().int()
    saa = sa.index_select(0, rows).cpu()
    sbb = sb.index_select(1, cols).cpu()
    expected = torch.zeros((len(rows), len(cols)), dtype=torch.float32)
    for g, start in enumerate(range(0, a.shape[1], 256)):
        dot = aa[:, start:start + 256] @ bb[start:start + 256]
        scale = saa[:, g, None] * sbb[g, None, :]
        # FP64 holds these bounded products/sums before rounding the FMA to FP32.
        expected = (dot.double() * scale.double() + expected.double()).float()
    return rows, cols, expected.to(torch.bfloat16)


def validate(out, ref, torch):
    rows, cols, expected = ref
    actual = out.index_select(0, rows).index_select(1, cols).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if not torch.isfinite(out).all().item():
        raise AssertionError('non-finite output')
    return dict(checked_rows=len(rows), checked_columns=len(cols),
                max_abs_error=(actual.float() - expected.float()).abs().max().item())


def run_case(spec, args, torch, grain, cuda, launch, bench):
    from triton.runtime.errors import OutOfResources

    m, n, k = (spec['shape'][axis] for axis in ('m', 'n', 'k'))
    torch.manual_seed(args.seed)
    a = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8).T
    sa = 0.001 + torch.rand((m, k // 256), device='cuda') * 0.009
    sb = 0.001 + torch.rand((k // 256, n), device='cuda') * 0.009
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref = reference(a, b, sa, sb, torch)
    native_ok = m % 64 == n % 64 == k % 256 == 0
    kwargs = dict(group_size=256, output_dtype=torch.bfloat16, scale_a=sa, scale_b=sb)
    defaults = {'triton_default': grain.get_kernel_config(a, b, backend='triton', **kwargs),
                'cute_default': grain.get_kernel_config(a, b, backend='cuda', **kwargs) if native_ok else None}
    auto = grain.get_kernel_config(a, b, backend='auto', **kwargs)

    def function(config):
        if config['backend'] == 'cuda':
            def run():
                return cuda.launch(a, b, sa, sb, out, config['config_id'])
        else:
            def run():
                launch(a, b, sa, sb, out, 256, config)
                return out
        return run

    candidates = triton_candidates()
    if native_ok:
        candidates += [dict(backend='cuda', config_id=i) for i, cfg in enumerate(cuda.CONFIGS)
                       if m % cfg['block_m'] == n % cfg['block_n'] == 0]
    for config in defaults.values():
        if config and canonical(config) not in candidates:
            candidates.append(canonical(config))
    funcs, checks, valid_candidates, skipped_candidates = [], [], [], []
    print(f'  Compile/check {len(candidates)} candidate configurations', flush=True)
    for config in candidates:
        fun = function(config)
        try:
            fun()
        except OutOfResources as error:
            if any(default and canonical(default) == config for default in defaults.values()):
                raise
            skipped_candidates.append(dict(config=config, reason=str(error)))
            continue
        checks.append(validate(out, ref, torch))
        funcs.append(fun)
        valid_candidates.append(config)
    candidates = valid_candidates
    samples = [[] for _ in candidates]
    for round_index in range(args.tune_rounds):
        print(f'  Tuning round {round_index + 1}/{args.tune_rounds}', flush=True)
        for step in range(len(candidates)):
            index = (step + round_index) % len(candidates)
            samples[index].append(bench(funcs[index], rep=args.tune_rep_ms, return_mode='median'))
    tuning = [dict(config=cfg, correctness=check, **summary(times, 2 * m * n * k))
              for cfg, check, times in zip(candidates, checks, samples)]
    configs = dict(defaults)
    for label, backend in [('triton_tuned', 'triton'), ('cute_tuned', 'cuda')]:
        choices = [entry for entry in tuning if entry['config']['backend'] == backend]
        configs[label] = min(choices, key=lambda entry: entry['median_ms'])['config'] if choices else None

    # Fresh input values for the final comparison; candidate selection stays fixed.
    torch.manual_seed(args.seed + 1)
    a.random_(-127, 128)
    b.T.random_(-127, 128)
    sa.uniform_(0.001, 0.01)
    sb.uniform_(0.001, 0.01)
    ref = reference(a, b, sa, sb, torch)
    selected = {name: function(config) for name, config in configs.items() if config}
    correctness = {}
    print('  Validate selected configurations and graph replay on fresh inputs', flush=True)
    for name, fun in selected.items():
        fun()
        correctness[name] = validate(out, ref, torch)
        saved_a = a.clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fun()
        graph.replay()
        validate(out, ref, torch)
        a.zero_()
        graph.replay()
        assert torch.count_nonzero(out).item() == 0, 'graph replay must read current activation'
        a.copy_(saved_a)
        graph.replay()
        validate(out, ref, torch)
        correctness[name]['graph_original_zero_restore'] = True
        del graph, saved_a
    for _ in range(args.warmup):
        for fun in selected.values():
            fun()
    torch.cuda.synchronize()
    measurements = {name: [] for name in selected}
    names = list(selected)
    orders = []
    for round_index in range(args.rounds):
        print(f'  Final timing round {round_index + 1}/{args.rounds}', flush=True)
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            measurements[name].append(bench(selected[name], rep=args.rep_ms, return_mode='median'))
    results = {name: summary(values, 2 * m * n * k) for name, values in measurements.items()}
    for name in METHODS:
        results.setdefault(name, None)
    case = dict(spec, timestamp_utc=stamp(), configs=configs, auto_config=auto, tuning=tuning,
                skipped_candidates=skipped_candidates,
                results=results, correctness=correctness, measurement_order=orders,
                native_supported=native_ok, group_size=256, output_dtype='bfloat16',
                default_speedup=None, tuned_speedup=None)
    if native_ok:
        case['default_speedup'] = results['triton_default']['median_ms'] / results['cute_default']['median_ms']
        case['tuned_speedup'] = results['triton_tuned']['median_ms'] / results['cute_tuned']['median_ms']
    else:
        case['native_unsupported_reason'] = 'Current CuTe kernel requires M and N multiples of 64; no padding was added.'
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=TOKENS)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--rep-ms', type=int, default=50)
    parser.add_argument('--tune-rounds', type=int, default=3)
    parser.add_argument('--tune-rep-ms', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int, help='smoke test: run only the first N distinct shapes')
    parser.add_argument('--resume', action='store_true', help='resume a saved incomplete sweep with the same kernels and settings')
    parser.add_argument('--max-new-cases', type=int, help='save and exit after this many additional shapes')
    args = parser.parse_args()
    if any(t <= 0 or t % 8 for t in args.tokens):
        parser.error('token counts must be positive multiples of 8 for the balanced-expert scenario')
    if len(set(args.tokens)) != len(args.tokens):
        parser.error('token counts must be distinct')
    if min(args.rounds, args.rep_ms, args.tune_rounds, args.tune_rep_ms, args.warmup) <= 0:
        parser.error('timing parameters must be positive')
    if args.max_new_cases is not None and args.max_new_cases <= 0:
        parser.error('max-new-cases must be positive')
    import torch
    import triton
    import grain_gemm as grain
    from grain_gemm.kernels import cuda
    from grain_gemm.kernels.triton import launch
    from triton.testing import do_bench_cudagraph

    assert torch.cuda.get_device_capability() == (12, 1), 'This benchmark targets the current GB10 native build'
    assert cuda.is_available(), 'Build the native library first'
    specs = workloads(args.tokens)
    if args.limit:
        specs = specs[:args.limit]
    package = Path(grain.__file__).parent
    report = dict(
        complete=False, started_at_utc=stamp(), gpu=torch.cuda.get_device_name(),
        tokens=args.tokens, group_size=256, output_dtype='bfloat16',
        expected_cases=len(specs), expected_workloads=specs,
        software=dict(python=platform.python_version(), torch=torch.__version__,
                      triton=triton.__version__, cuda=torch.version.cuda),
        model=dict(hidden=1536, layers=24, query_heads=24, kv_heads=4, head_dim=128,
                   vocabularies=[151936, 262144], routed_experts=32, top_k=4,
                   expert_intermediate=1024, shared_intermediate=1024,
                   activation_assumption='gated SiLU / SwiGLU; separate and fused projections are alternatives',
                   token_assumption='T is total prefill tokens (batch times length for uniform batches); expert M=T/8 assumes balanced routing',
                   attention='20 SWA layers, 4 global layers; QK/PV attention matmuls are outside this linear-projection benchmark'),
        timing=dict(rounds=args.rounds, rep_ms=args.rep_ms, tune_rounds=args.tune_rounds,
                    tune_rep_ms=args.tune_rep_ms, warmup_calls=args.warmup,
                    method='triton.testing.do_bench_cudagraph', statistic='median of per-round medians',
                    order='rotating per round', cache_policy='repeated inputs; no cache flush',
                    scope='Preallocated-output GPU GEMM only; includes INT32 group reduction, FP32 scaling/accumulation, BF16 output; excludes input quantization, compilation, tuning, host dispatch and allocation',
                    tuning='Select each backend separately from existing candidates; final measurement uses fresh data and a separate timing pass'),
        inputs=dict(quantization='synthetic already-quantized INT8, uniform integers [-127,127]; no model weights used',
                    scales='independent FP32 uniform [0.001,0.01)', tuning_seed=args.seed, final_seed=args.seed+1,
                    a_layout='row-major', b_layout='column-major view of contiguous [N,K]', scales_layout='contiguous'),
        source=dict(git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'], text=True).strip(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    kernel_hashes={str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (package/'gemm.py', package/'kernels/triton.py', package/'kernels/cuda.py', package/'kernels/csrc/grain_cute.cu')},
                    native_build=json.loads((package/'kernels/_native/build.json').read_text()),
                    native_library_sha256=hashlib.sha256((package/'kernels/_native/libgrain_cuda.so').read_bytes()).hexdigest()),
        native_configurations=list(cuda.CONFIGS), triton_candidates=triton_candidates(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads(args.output.read_text())
        assert previous['complete'] is False, 'The saved report is already complete'
        for key in ('tokens', 'group_size', 'output_dtype', 'expected_cases', 'expected_workloads',
                    'timing', 'inputs', 'software', 'gpu', 'native_configurations', 'triton_candidates'):
            assert previous[key] == report[key], f'Resume metadata changed: {key}'
        for key in ('git_commit', 'kernel_hashes', 'native_library_sha256'):
            assert previous['source'][key] == report['source'][key], f'Resume source changed: {key}'
        for index, case in enumerate(previous['cases']):
            assert case['shape'] == specs[index]['shape'] and case['uses'] == specs[index]['uses']
        previous.setdefault('resume_sessions', []).append(dict(
            timestamp_utc=stamp(), completed_cases=len(previous['cases']),
            script_sha256=report['source']['script_sha256'],
            boot_time=subprocess.check_output(['uptime', '-s'], text=True).strip(),
            note='Resumed saved checkpoint; kernels, software versions and measurement settings verified unchanged'))
        previous.pop('failure', None)
        report = previous
    save(report, args.output)
    completed_before = len(report['cases'])
    for index, spec in enumerate(specs, 1):
        if index <= len(report['cases']):
            continue
        print(f'[{index}/{len(specs)}] {spec["shape"]} {spec["uses"][0]["label"]}', flush=True)
        try:
            case = run_case(spec, args, torch, grain, cuda, launch, do_bench_cudagraph)
        except Exception as error:
            report['failure'] = dict(shape=spec['shape'], type=type(error).__name__, message=str(error))
            save(report, args.output)
            raise
        report['cases'].append(case)
        save(report, args.output)
        display = ', '.join(f'{name}={result["median_ms"]*1000:.2f}us' for name, result in case['results'].items() if result)
        print('  ' + display, flush=True)
        if case['native_supported']:
            print(f'  CuTe speedup default={case["default_speedup"]:.3f}x tuned={case["tuned_speedup"]:.3f}x', flush=True)
        # Keep the allocator cache between cases; allocation is outside timing.
        if args.max_new_cases and len(report['cases']) - completed_before >= args.max_new_cases:
            break
    if len(report['cases']) != len(specs):
        print(f'Saved checkpoint: {len(report["cases"])}/{len(specs)} shapes; sweep remains incomplete', flush=True)
        return
    report['complete'] = True
    report['completed_at_utc'] = stamp()
    save(report, args.output)
    print(f'Saved {len(specs)} distinct shapes to {args.output}', flush=True)


if __name__ == '__main__':
    main()
