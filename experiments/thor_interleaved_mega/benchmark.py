#!/usr/bin/env python3
"""Compare full eager public interleaved INT8 MoE and the explicit Thor Mega experiment.

Input quantization and weight preparation are outside timing. This script never
changes clocks and runs no GPU work when imported or invoked with --help.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    with output.with_suffix('.csv').open('w', newline='') as stream:
        fields = ['tokens', 'method', 'median_ms', 'effective_tops', 'current_over_method']
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in report['cases']:
            for method, result in case.get('results', {}).items():
                writer.writerow(dict(tokens=case['tokens'], method=method,
                    **{key: result[key] for key in fields[2:]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--tokens', nargs='+', type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--output', type=Path, default=HERE/'build/benchmark.json')
    parser.add_argument('--smoke-only', action='store_true', help='Exact output checks and warmup; no timing')
    args = parser.parse_args()
    if min(args.rounds, args.samples) < 1 or min(args.tokens) < 1 or len(set(args.tokens)) != len(args.tokens):
        parser.error('Require positive rounds/samples and distinct positive token counts')
    if args.output.suffix != '.json':
        parser.error('--output must end in .json')
    report = dict(complete=False, success=False, cases=[], tokens=args.tokens,
        experts=32, hidden=1536, intermediate=1024, top_k=4, seed=2026,
        rounds=args.rounds, samples=args.samples, smoke_only=args.smoke_only,
        timing='Full eager API CUDA events, no graph; validation, routing, CPU statistics read and per-call allocations included',
        preparation='Same prequantized original tokens and physically interleaved weights/scales; input quantization and weight permutation outside timing',
        throughput='Useful projection operations 6*T*R*H*I / elapsed seconds; INT8 TOPS; padding and activation arithmetic excluded',
        thermal='Read-only GPU1197MHz/EMC2750MHz min/max/current; >=62C coolto54C, abort>=70C; 256MB cache flush outside each sample',
        comparison='Two INT8 implementations only; no FP8 baseline or quality comparison')
    guard = None
    before = None
    save(report, args.output)
    try:
        sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmarks'), str(HERE)]
        import torch
        from grain_gemm import moe_ffn
        from local_ctypes_api import moe_ffn_mega, build_metadata
        from bench_moe_interleaved import (Guard, hash_package, interleave_weight,
            prepare_weight, quantize_rows, difference)
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (11, 0):
            raise RuntimeError('Require Thor SM110')
        guard = Guard(torch)
        report['state_before'] = guard.state()
        metadata = build_metadata(args.library)
        report['build'] = metadata
        def provenance():
            return dict(experiment={p.name: sha(p) for p in sorted(HERE.iterdir())
                if p.suffix in ('.py', '.cu', '.cuh', '.hpp', '.h')},
                library=sha(metadata['library']), metadata=sha(Path(metadata['library']).with_suffix('.json')),
                current=hash_package(ROOT/'src/grain_gemm'),
                benchmark_helper=sha(ROOT/'benchmarks/bench_moe_interleaved.py'))
        before = provenance()
        report['provenance_before'] = before
        report['device'] = torch.cuda.get_device_name()
        with torch.inference_mode():
            torch.manual_seed(2026)
            e, h, i, r = 32, 1536, 1024, 4
            w1 = guard.call(lambda: torch.randn((e, 2*i, h), device='cuda', dtype=torch.bfloat16) / h**.5)
            w2 = guard.call(lambda: torch.randn((e, h, i), device='cuda', dtype=torch.bfloat16) / i**.5)
            q1, s1 = guard.call(lambda: prepare_weight(w1, torch))
            q2, s2 = guard.call(lambda: prepare_weight(w2, torch))
            qi, si = guard.call(lambda: interleave_weight(q1, s1))
            del w1, w2, q1, s1
            for tokens in args.tokens:
                case = dict(tokens=tokens, results={})
                report['cases'].append(case)
                save(report, args.output)
                torch.manual_seed(2026 + tokens)
                x = guard.call(lambda: torch.randn((tokens, h), device='cuda', dtype=torch.bfloat16))
                qx, sx = guard.call(lambda: quantize_rows(x, torch))
                del x
                ids = guard.call(lambda: torch.randn((tokens, e), device='cuda').topk(r, -1).indices.contiguous())
                def make_scores():
                    scores = torch.rand((tokens, r), device='cuda')
                    return (scores / scores.sum(-1, keepdim=True)).bfloat16().float().contiguous()
                scores = guard.call(make_scores)
                keywords = dict(scale_x=sx, scale_gate_up=si, scale_down=s2,
                    output_dtype=torch.bfloat16, backend='auto', gate_up_layout='interleaved')
                methods = dict(current=lambda: moe_ffn(qx, qi, q2, scores, ids, **keywords),
                    mega=lambda: moe_ffn_mega(qx, qi, q2, scores, ids, **keywords, library=args.library))
                # These untimed calls also warm both complete implementations.
                outputs = {name: guard.call(lambda fn=fn: fn().cpu()) for name, fn in methods.items()}
                case['checks'] = dict(output_bits_equal=bool(torch.equal(
                    outputs['current'].view(torch.int16), outputs['mega'].view(torch.int16))),
                    all_finite=all(bool(torch.isfinite(value).all()) for value in outputs.values()))
                case['difference'] = difference(outputs['mega'], outputs['current'], torch)
                if not all(case['checks'].values()):
                    raise AssertionError(case['checks'])
                reference = outputs['current']
                del outputs
                if not args.smoke_only:
                    case['results'] = guard.measure(methods, rounds=args.rounds,
                        samples=args.samples, graph_calls=0)
                    current_ms = case['results']['current']['median_ms']
                    for result in case['results'].values():
                        result['effective_tops'] = 6*tokens*r*h*i / (result['median_ms']*1e9)
                        result['current_over_method'] = current_ms / result['median_ms']
                case['post_timing_checks'] = {}
                for name, fn in methods.items():
                    repeated = guard.call(lambda fn=fn: fn().cpu())
                    checks = dict(output_bits_equal=bool(torch.equal(
                        repeated.view(torch.int16), reference.view(torch.int16))),
                        all_finite=bool(torch.isfinite(repeated).all()))
                    case['post_timing_checks'][name] = checks
                    if not all(checks.values()):
                        raise AssertionError({'method': name, 'post_timing': checks})
                del repeated, reference
                print(json.dumps(case), flush=True)
                save(report, args.output)
            report.update(complete=True, success=True)
    except Exception as error:
        report['failure'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        print(report['failure']['traceback'], file=sys.stderr, flush=True)
    finally:
        if before is not None:
            try:
                report['provenance_after'] = provenance()
                report['provenance_unchanged'] = report['provenance_after'] == before
                if not report['provenance_unchanged']:
                    report.update(complete=False, success=False)
            except Exception as error:
                report.update(complete=False, success=False, provenance_error=repr(error))
        if guard is not None:
            report['clock_temperature_snapshots'] = guard.states
            report['maximum_temperature_c'] = max((s['temperature_c'] for s in guard.states), default=None)
        save(report, args.output)
    print(f'success={report["success"]}; output={args.output}', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
