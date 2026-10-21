#!/usr/bin/env python3
"""Compare split and physically interleaved gate/up MoE weights on Thor.

Example (this script never changes clocks):
  PYTHONPATH=src python benchmarks/bench_moe_interleaved.py \
      --before-package-dir /tmp/grain_gemm_before_interleave \
      --tokens 4096 --quick --output /tmp/moe_interleaved_pilot.json

The frozen package must be the immediately preceding implementation. Weight
permutation and input quantization occur outside timing. Full public calls use
CUDA events in eager mode; isolated fused G1/SwiGLU/quantization uses at most two
calls per graph replay. Both useful-operation numerators exclude padding.
"""

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('before_split', 'current_split', 'interleaved')
SOURCE_SUFFIXES = {'.py', '.cu', '.cuh', '.h', '.hpp', '.json'}
GPU_HZ, EMC_HZ = 1197000000, 2750000000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_package(directory):
    directory = Path(directory).resolve()
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.rglob('*'))
            if path.is_file() and (path.suffix in SOURCE_SUFFIXES or path.suffix == '.so')
            and '__pycache__' not in path.parts}


def load_package(directory, alias):
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location(
        alias, directory / '__init__.py', submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load package {directory}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def save(report, output):
    report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    fields = ['tokens', 'scope', 'method', 'median_ms', 'effective_tops',
              'before_over_method', 'current_split_over_method', 'round_medians_ms']
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in report['cases']:
            for scope, results in case.get('results', {}).items():
                before = results.get('before_split', {}).get('median_ms')
                split = results.get('current_split', {}).get('median_ms')
                for method, result in results.items():
                    writer.writerow(dict(tokens=case['tokens'], scope=scope, method=method,
                        median_ms=result['median_ms'], effective_tops=result['effective_tops'],
                        before_over_method=before / result['median_ms'] if before else None,
                        current_split_over_method=split / result['median_ms'] if split else None,
                        round_medians_ms=json.dumps(result['round_medians_ms'])))


class Guard:
    """Read-only clocks; short workloads separated by temperature checks."""
    def __init__(self, torch):
        self.torch = torch
        self.paths = [path / 'temp' for path in Path('/sys/class/thermal').glob('thermal_zone*')
                      if 'gpu' in (path / 'type').read_text().lower()]
        if not self.paths:
            raise RuntimeError('No GPU thermal sensor found')
        self.states = []
        self.flush = torch.empty(256_000_000, device='cuda', dtype=torch.uint8)

    def state(self):
        clocks = {f'{device}_{name}': int((Path('/sys/class/devfreq') / device / name).read_text())
                  for device in ('gpu-gpc-0', 'bwmgr')
                  for name in ('min_freq', 'max_freq', 'cur_freq')}
        temperature = max(int(path.read_text()) / 1000 for path in self.paths)
        value = dict(clocks=clocks, temperature_c=temperature)
        self.states.append(value)
        if any(frequency != (GPU_HZ if name.startswith('gpu') else EMC_HZ)
               for name, frequency in clocks.items()):
            raise RuntimeError(f'Require fixed GPU1197MHz and EMC2750MHz: {value}')
        if temperature >= 70:
            raise RuntimeError(f'Temperature reached the70C abort threshold: {value}')
        return value

    def cool(self):
        if self.state()['temperature_c'] < 62:
            return
        self.torch.cuda.synchronize()
        started = time.monotonic()
        while self.state()['temperature_c'] > 54:
            if time.monotonic() - started > 120:
                raise RuntimeError('Cooling to54C timed out')
            time.sleep(.25)

    def call(self, function):
        self.cool()
        result = function()
        self.torch.cuda.synchronize()
        self.state()
        return result

    def measure(self, functions, *, rounds, samples, graph_calls):
        torch = self.torch
        graphs = {}
        if graph_calls:
            for method, function in functions.items():
                self.call(function)
                graph = torch.cuda.CUDAGraph()
                self.cool()
                with torch.cuda.graph(graph):
                    for _ in range(graph_calls):
                        function()
                self.call(graph.replay)
                graphs[method] = graph
        starts = torch.cuda.Event(enable_timing=True)
        ends = torch.cuda.Event(enable_timing=True)
        starts.record(); ends.record(); ends.synchronize()
        names = tuple(functions)
        results = {name: dict(round_medians_ms=[], samples=[]) for name in names}
        for rd in range(rounds):
            values = {name: [] for name in names}
            for sample in range(samples):
                offset = (rd + sample) % len(names)
                order = names[offset:] + names[:offset]
                if sample % 2:
                    order = order[::-1]
                for method in order:
                    self.cool()
                    self.flush.zero_()
                    torch.cuda.synchronize()
                    before = self.state()
                    starts.record()
                    if graph_calls:
                        graphs[method].replay()
                        output = None
                    else:
                        output = functions[method]()
                    ends.record(); ends.synchronize()
                    latency = starts.elapsed_time(ends) / (graph_calls or 1)
                    after = self.state()
                    if latency <= 0:
                        raise RuntimeError('Nonpositive event duration')
                    values[method].append(latency)
                    results[method]['samples'].append(dict(round=rd, sample=sample,
                        order=list(order), ms=latency, before=before, after=after))
                    del output
            for method in names:
                results[method]['round_medians_ms'].append(statistics.median(values[method]))
        for result in results.values():
            result['median_ms'] = statistics.median(result['round_medians_ms'])
        return results


def quantize_rows(x, torch):
    groups = x.float().reshape(x.shape[0], -1, 256)
    scales = groups.abs().amax(-1).clamp_min(1e-12) / 127
    quantized = (groups / scales[..., None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(x.shape), scales


def prepare_weight(weight, torch):
    e, n, k = weight.shape
    q, scale = quantize_rows(weight.reshape(e * n, k), torch)
    return q.reshape(e, n, k).transpose(-2, -1), scale.reshape(e, n, k // 256).transpose(-2, -1).contiguous()


def interleave_weight(qweight, scales):
    """Preserve GEMM logical shape/stride; permute physical Linear output rows."""
    e, h, twice_i = qweight.shape
    i = twice_i // 2
    linear = qweight.transpose(-2, -1)
    packed = linear.view(e, 2, i, h).transpose(1, 2).contiguous().view(e, twice_i, h)
    scale_packed = scales.view(e, h // 256, 2, i).transpose(2, 3).contiguous().view(e, h // 256, twice_i)
    return packed.transpose(-2, -1), scale_packed


def difference(value, reference, torch):
    value, reference = value.cpu().float(), reference.cpu().float()
    error = value - reference
    return dict(relative_l2=float(error.norm() / reference.norm().clamp_min(1e-12)),
                max_abs=float(error.abs().max()) if error.numel() else 0.)


def cpu_reference(x, ids, scores, w1, w2, s1, s2, sx, torch, *, token_mask=None):
    """Independent INT32 group dots, split weights, explicit BF16 round points."""
    x, ids, scores, w1, w2, s1, s2, sx = (
        value.cpu() for value in (x, ids, scores, w1, w2, s1, s2, sx))
    mask = token_mask.cpu() if token_mask is not None else None
    def project(a, sa, b, sb):
        output = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
        for group, start in enumerate(range(0, a.shape[1], 256)):
            dot = a[:, start:start + 256].int() @ b[start:start + 256].int()
            scale = sa[:, group, None] * sb[group, None, :]
            output = (output.double() + dot.double() * scale.double()).float()
        return output.bfloat16().float()
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        if mask is not None and not mask[token]:
            continue
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = project(x[token:token + 1], sx[token:token + 1], w1[expert], s1[expert]).chunk(2, -1)
            hidden = torch.nn.functional.silu(gate) * up
            qh, sh = quantize_rows(hidden, torch)
            value = project(qh, sh, w2[expert], s2[expert])
            output[token] += value[0] * scores[token, slot]
    return output.bfloat16()


def capture_stage(package, function, guard, *, projection_name='moe_gather'):
    """Capture the actual public fused launch and G2 inputs outside timing."""
    implementation = importlib.import_module(package.__name__ + '.moe')
    if projection_name not in ('moe_gather', 'moe_interleaved'):
        raise ValueError('Unexpected first-projection bridge: ' + projection_name)
    fused = importlib.import_module(package.__name__ + '.kernels.' + projection_name)
    launch = fused._launch_prevalidated
    native = implementation._native_gemm
    data = dict(fused_calls=[], native_calls=[])
    def record_fused(*args, **kwargs):
        data['fused_calls'].append(dict(bridge=projection_name, x_shape=list(args[0].shape), weight_shape=list(args[1].shape),
            out_shape=list(args[7].shape), keywords=dict(kwargs)))
        data['stage_function'] = lambda args=args, kwargs=kwargs: launch(*args, **kwargs)
        return launch(*args, **kwargs)
    def record_native(*args, **kwargs):
        data['native_calls'].append(dict(a_shape=list(args[0].shape), weight_shape=list(args[1].shape), backend=args[5]))
        data['qh'], data['sh'] = args[0], args[2]
        data['counts'] = kwargs['counts']
        return native(*args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(fused, '_launch_prevalidated', record_fused))
        stack.enter_context(patch.object(implementation, '_native_gemm', record_native))
        data['output'] = guard.call(function)
    if len(data['fused_calls']) != 1 or len(data['native_calls']) != 1:
        raise AssertionError('Expected one original-token fused G1 and one native G2: ' + str(data['native_calls']))
    data['qh_reference'], data['sh_reference'] = data['qh'].clone(), data['sh'].clone()
    return data


def run_edge_cases(before, current, weights, guard, torch):
    q1, s1, qi, si, q2, s2 = weights
    e, h, twice_i = q1.shape
    r = 4
    results = []
    for name, tokens, mode in (
            ('nondivisible_balanced', 257, 'cyclic'), ('duplicate_expert_masked', 65, 'duplicate'),
            ('skewed_empty_experts', 65, 'skew'), ('all_masked_invalid_ids', 17, 'all_masked'),
            ('zero_tokens', 0, 'zero')):
        torch.manual_seed(1701 + tokens)
        x = torch.randint(-127, 128, (tokens, h), device='cuda', dtype=torch.int8)
        sx = torch.full((tokens, h // 256), .01, device='cuda', dtype=torch.float32)
        token = torch.arange(tokens, device='cuda')
        ids = ((token[:, None] * r + torch.arange(r, device='cuda')) % e).long().contiguous()
        scores = torch.full((tokens, r), 1 / r, device='cuda', dtype=torch.float32)
        mask = None
        if mode == 'duplicate':
            ids = (token[:, None] % e).expand(-1, r).contiguous()
            mask = (token % 3 != 1).contiguous()
            ids[~mask] = -1
        elif mode == 'skew':
            ids = torch.arange(r, device='cuda')[None, :].expand(tokens, -1).contiguous()
        elif mode == 'all_masked':
            mask = torch.zeros(tokens, device='cuda', dtype=torch.bool)
            ids.fill_(e + 7)
        for backend in ('cutlass', 'triton'):
            outputs = {}
            for method, package, w1, sf1, layout in (
                    ('before_split', before, q1, s1, None),
                    ('current_split', current, q1, s1, 'split'),
                    ('interleaved', current, qi, si, 'interleaved')):
                keywords = dict(scale_x=sx, scale_gate_up=sf1, scale_down=s2,
                                token_mask=mask, backend=backend)
                if layout is not None:
                    keywords['gate_up_layout'] = layout
                outputs[method] = guard.call(lambda package=package, w1=w1, keywords=keywords:
                    package.moe_ffn(x, w1, q2, scores, ids, **keywords))
            equality = {method: bool(torch.equal(output.view(torch.int16), outputs['before_split'].view(torch.int16)))
                        for method, output in outputs.items()}
            finite = all(bool(torch.isfinite(output).all()) for output in outputs.values())
            masked_zero = mask is None or all(bool((output[~mask] == 0).all()) for output in outputs.values())
            if not all(equality.values()) or not finite or not masked_zero:
                raise AssertionError(f'Edge case failed: {name}, {backend}, {equality}')
            item = dict(name=name, tokens=tokens, backend=backend, output_bits_equal=equality,
                        all_finite=finite, masked_output_zero=masked_zero)
            if tokens and mode != 'all_masked':
                selected = torch.tensor([0, tokens - 1], device='cuda')
                reference = cpu_reference(x[selected], ids[selected], scores[selected], q1, q2, s1, s2,
                    sx[selected], torch, token_mask=mask[selected] if mask is not None else None)
                metrics = difference(outputs['interleaved'][selected], reference, torch)
                if metrics['relative_l2'] > .03 or metrics['max_abs'] > .05:
                    raise AssertionError(f'Edge CPU reference failed: {name}: {metrics}')
                item['independent_quantized_reference'] = metrics
            results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-package-dir', type=Path, required=True)
    parser.add_argument('--current-package-dir', type=Path, default=ROOT / 'src/grain_gemm')
    parser.add_argument('--tokens', nargs='+', type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--stage-graph-calls', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--skip-edge-checks', action='store_true')
    parser.add_argument('--quick', action='store_true', help='One round/sample/warm call, no edge cases, one call per stage graph')
    parser.add_argument('--smoke-only', action='store_true', help='Correctness and warmup only, no timed repetitions')
    parser.add_argument('--output', type=Path, default=Path('/tmp/moe_interleaved.json'))
    args = parser.parse_args()
    if not args.tokens or min(args.tokens) < 2 or len(set(args.tokens)) != len(args.tokens):
        parser.error('Require distinct token counts>=2')
    if min(args.rounds, args.samples) < 1 or args.warmup < 0 or args.output.suffix != '.json':
        parser.error('Require positive rounds/samples, nonnegative warmup, and .json output')
    if args.quick:
        args.rounds = args.samples = args.warmup = args.stage_graph_calls = 1
        args.skip_edge_checks = True
    args.before_package_dir = args.before_package_dir.resolve()
    args.current_package_dir = args.current_package_dir.resolve()
    if args.before_package_dir == args.current_package_dir:
        parser.error('Before and current package directories must differ')
    for directory in (args.before_package_dir, args.current_package_dir):
        if not (directory / '__init__.py').is_file():
            parser.error('Missing package __init__.py in ' + str(directory))
    report = dict(complete=False, success=False, stage='initialize', cases=[], edge_checks=[],
        seed=2026, experts=32, hidden=1536, intermediate=1024, top_k=4, tokens=args.tokens,
        before_package=str(args.before_package_dir), current_package=str(args.current_package_dir),
        rounds=args.rounds, samples=args.samples, warmup=args.warmup, quick=args.quick,
        timing=dict(full_ffn='Eager full public call CUDA events; no CUDA Graph',
                    gate_up=f'Actual public fused G1+SwiGLU+G256 quantization; {args.stage_graph_calls}calls/graph' if args.stage_graph_calls else 'Actual public fused stage, eager events'),
        operation_numerator=dict(full_ffn='6*T*TopK*H*I', gate_up='4*T*TopK*H*I; excludes activation/quant arithmetic'),
        weight_layout=dict(split='g0,g1,...,u0,u1,...', interleaved='g0,u0,g1,u1,...',
            logical_weight='[E,H,2I], stride=(H*2I,1,H)', scale='[E,H/256,2I], contiguous; same output-column permutation'),
        scope='Same quantized input, random weight values, route IDs and scores; weight/input preparation outside timing; useful operations exclude padding',
        thermal='Read-only1197MHz GPU/2750MHz EMC min/max/current each sample; >=62C coolto54C, abort>=70C; one warmcall default;256MB L2 flush outside every timed sample',
        baseline='Frozen immediately preceding original-token gather fused G1+SwiGLU+quantization; full GU tensor already absent',
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    save(report, args.output)
    provenance_before = None
    guard = None
    try:
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (11, 0):
            raise RuntimeError('Require Thor SM110')
        guard = Guard(torch)
        report['all_clock_temperature_snapshots'] = guard.states
        report['state_before'] = guard.state()
        provenance_before = dict(before=hash_package(args.before_package_dir), current=hash_package(args.current_package_dir))
        report['provenance_before'] = provenance_before
        frozen_manifest = args.before_package_dir.with_name(args.before_package_dir.name + '_manifest.json')
        if frozen_manifest.is_file():
            entries = json.loads(frozen_manifest.read_text())
            manifest_root = args.before_package_dir
            for relative, expected in entries.items():
                target = (manifest_root / relative).resolve()
                if not target.is_relative_to(manifest_root) or not target.is_file() or sha(target) != expected:
                    raise RuntimeError('Frozen baseline differs from its manifest: ' + relative)
            report['frozen_before_manifest'] = dict(path=str(frozen_manifest), sha256=sha(frozen_manifest), verified_files=len(entries))
        before = load_package(args.before_package_dir, 'grain_gemm_before_interleaved_benchmark')
        current = load_package(args.current_package_dir, 'grain_gemm_current_interleaved_benchmark')
        packages = dict(before_split=before, current_split=current, interleaved=current)
        report['software'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  device=torch.cuda.get_device_name())
        report['checkout'] = dict(head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            status=subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True))
        with torch.inference_mode():
            torch.manual_seed(2026)
            e, h, i, r = 32, 1536, 1024, 4
            report['stage'] = 'prepare_weights'
            save(report, args.output)
            w1 = guard.call(lambda: (torch.randn((e, 2 * i, h), device='cuda', dtype=torch.bfloat16) / h**.5).contiguous())
            w2 = guard.call(lambda: (torch.randn((e, h, i), device='cuda', dtype=torch.bfloat16) / i**.5).contiguous())
            q1, s1 = guard.call(lambda: prepare_weight(w1, torch))
            q2, s2 = guard.call(lambda: prepare_weight(w2, torch))
            qi, si = guard.call(lambda: interleave_weight(q1, s1))
            inverse_q = qi.transpose(-2, -1).view(e, i, 2, h).transpose(1, 2).contiguous().view(e, 2 * i, h).transpose(-2, -1)
            inverse_s = si.view(e, h // 256, i, 2).transpose(2, 3).contiguous().view(e, h // 256, 2 * i)
            if not torch.equal(inverse_q, q1) or not torch.equal(inverse_s.view(torch.int32), s1.view(torch.int32)):
                raise AssertionError('Weight or scale interleaving permutation failed exact inverse')
            report['weight_permutation_exact_inverse'] = True
            del inverse_q, inverse_s, w1, w2
            if not args.skip_edge_checks:
                report['stage'] = 'edge_correctness'
                save(report, args.output)
                report['edge_checks'] = run_edge_cases(before, current, (q1, s1, qi, si, q2, s2), guard, torch)
                save(report, args.output)
            for tokens in args.tokens:
                report['stage'] = f'T{tokens}:prepare'
                case = dict(tokens=tokens, results={}, checks={})
                report['cases'].append(case)
                save(report, args.output)
                torch.manual_seed(2026 + tokens)
                x = guard.call(lambda: torch.randn((tokens, h), device='cuda', dtype=torch.bfloat16))
                ids = guard.call(lambda: torch.randn((tokens, e), device='cuda').topk(r, -1).indices.contiguous())
                scores = guard.call(lambda: torch.rand((tokens, r), device='cuda'))
                scores = (scores / scores.sum(-1, keepdim=True)).bfloat16().float().contiguous()
                qx, sx = guard.call(lambda: quantize_rows(x, torch))
                del x
                functions = {}
                for method in METHODS:
                    package = packages[method]
                    weight, sf = (qi, si) if method == 'interleaved' else (q1, s1)
                    keywords = dict(scale_x=sx, scale_gate_up=sf, scale_down=s2, backend='auto')
                    if method != 'before_split':
                        keywords['gate_up_layout'] = 'interleaved' if method == 'interleaved' else 'split'
                    functions[method] = lambda package=package, weight=weight, keywords=keywords: package.moe_ffn(
                        qx, weight, q2, scores, ids, **keywords)
                report['stage'] = f'T{tokens}:correctness'
                captures = {method: capture_stage(packages[method], function, guard,
                                projection_name='moe_interleaved' if method == 'interleaved' else 'moe_gather')
                            for method, function in functions.items()}
                first = captures['before_split']
                capacity = first['qh'].shape[1]
                counts = first['counts']
                expected_counts = torch.bincount(ids.flatten(), minlength=e).int()
                padding = torch.arange(capacity, device='cuda')[None, :] >= counts[:, None]
                zero_sf = (torch.tensor(1.e-12, dtype=torch.float32) / torch.tensor(127., dtype=torch.float32)).to('cuda')
                case.update(capacity=capacity, expert_counts=counts.cpu().tolist(),
                    padded_rows=int(padding.sum()), actual_dispatch={method: dict(fused=data['fused_calls'], native=data['native_calls'])
                                                                 for method, data in captures.items()})
                for method, data in captures.items():
                    checks = dict(qh_full_bits_equal=bool(torch.equal(data['qh'], first['qh'])),
                        sh_full_bits_equal=bool(torch.equal(data['sh'].view(torch.int32), first['sh'].view(torch.int32))),
                        output_full_bits_equal=bool(torch.equal(data['output'].view(torch.int16), first['output'].view(torch.int16))),
                        counts_equal=bool(torch.equal(data['counts'], expected_counts)),
                        padding_q_zero=bool((data['qh'][padding] == 0).all()),
                        padding_scale_bits_equal=bool((data['sh'][padding].view(torch.int32) == zero_sf.view(torch.int32)).all()),
                        all_finite=bool(torch.isfinite(data['output']).all()))
                    case['checks'][method] = checks
                    if not all(checks.values()):
                        raise AssertionError(f'{method}: full QH/SH/output/count/padding equivalence failed: {checks}')
                reference = cpu_reference(qx[:2], ids[:2], scores[:2], q1, q2, s1, s2, sx[:2], torch)
                metrics = difference(first['output'][:2], reference, torch)
                case['independent_cpu_quantized_reference'] = metrics
                if metrics['relative_l2'] > .03 or metrics['max_abs'] > .05:
                    raise AssertionError('Independent CPU reference failed: ' + str(metrics))
                for _ in range(args.warmup):
                    for function in functions.values():
                        guard.call(function)
                stages = {method: data['stage_function'] for method, data in captures.items()}
                if not args.smoke_only:
                    report['stage'] = f'T{tokens}:full_ffn_timing'
                    save(report, args.output)
                    case['results']['full_ffn'] = guard.measure(functions, rounds=args.rounds, samples=args.samples, graph_calls=0)
                    for result in case['results']['full_ffn'].values():
                        result['effective_tops'] = 6 * tokens * r * h * i / (result['median_ms'] * 1e9)
                    report['stage'] = f'T{tokens}:gate_up_timing'
                    save(report, args.output)
                    case['results']['gate_up'] = guard.measure(stages, rounds=args.rounds, samples=args.samples,
                                                              graph_calls=args.stage_graph_calls)
                    for result in case['results']['gate_up'].values():
                        result['effective_tops'] = 4 * tokens * r * h * i / (result['median_ms'] * 1e9)
                case['post_timing_checks'] = {}
                for method, function in functions.items():
                    output = guard.call(function)
                    data = captures[method]
                    guard.call(data['stage_function'])
                    checks = dict(full_output_bits_equal=bool(torch.equal(output.view(torch.int16), first['output'].view(torch.int16))),
                        stage_qh_bits_equal=bool(torch.equal(data['qh'], data['qh_reference'])),
                        stage_sh_bits_equal=bool(torch.equal(data['sh'].view(torch.int32), data['sh_reference'].view(torch.int32))))
                    case['post_timing_checks'][method] = checks
                    if not all(checks.values()):
                        raise AssertionError(f'{method}: post-timing output/stage changed: {checks}')
                if not args.smoke_only:
                    print(f'T{tokens}: ' + '; '.join(scope + ': ' + ', '.join(
                        f'{method}={result["median_ms"]:.6f}ms ({result["effective_tops"]:.2f}TOPS)'
                        for method, result in results.items()) for scope, results in case['results'].items()), flush=True)
                else:
                    print(f'T{tokens}: all QH/SH/output bits and CPU reference passed', flush=True)
                save(report, args.output)
                del functions, captures, first, padding, counts, expected_counts, qx, sx, ids, scores, output, data, stages
        report.update(complete=True, success=True, stage='finished')
    except Exception as error:
        report['failure'] = dict(stage=report['stage'], type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        print(f'FAILED {report["stage"]}: {error}', file=sys.stderr, flush=True)
    finally:
        if provenance_before is not None:
            try:
                after = dict(before=hash_package(args.before_package_dir), current=hash_package(args.current_package_dir))
                report['provenance_after'] = after
                manifest = report.get('frozen_before_manifest')
                manifest_unchanged = manifest is None or sha(manifest['path']) == manifest['sha256']
                report['provenance_unchanged'] = (after == provenance_before
                    and sha(__file__) == report['script']['sha256'] and manifest_unchanged)
                if not report['provenance_unchanged']:
                    report.update(success=False, complete=False)
                    report['provenance_error'] = 'Package or script changed during run'
            except Exception as error:
                report.update(success=False, complete=False, provenance_error=repr(error))
        if guard is not None:
            report['all_clock_temperature_snapshots'] = guard.states
            if guard.states:
                report['maximum_temperature_c'] = max(value['temperature_c'] for value in guard.states)
        save(report, args.output)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
