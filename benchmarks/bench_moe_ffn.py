"""Measure the complete INT8 MoE FFN against an eager BF16/FP16 expert reference.

Both paths include token routing, both expert projections, SwiGLU, and weighted
combination. INT8 timings include activation quantization but exclude one-time
weight quantization. The default BF16 baseline is a readable per-expert PyTorch
loop, not a tuned fused BF16 MoE kernel. Results must be interpreted accordingly.

Example: PYTHONPATH=src python benchmarks/bench_moe_ffn.py --tokens 128 512 2048 \
    --output benchmarks/results/thor/moe_ffn.json
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--backend", choices=["auto", "triton", "cuda", "cutlass"], default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-tokens", type=int, default=2)
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--max-temperature", type=float, default=85)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/thor/moe_ffn.json")
    args = parser.parse_args()
    if not args.tokens or min(args.tokens) <= 0 or len(set(args.tokens)) != len(args.tokens):
        parser.error("tokens must be distinct positive integers")
    if min(args.experts, args.hidden, args.intermediate, args.top_k) <= 0:
        parser.error("expert count, hidden/intermediate dimensions and top-k must be positive")
    if args.hidden % 256 or args.intermediate % 256:
        parser.error("hidden and intermediate dimensions must be multiples of 256")
    if args.top_k > args.experts:
        parser.error("top-k must not exceed the expert count")
    if min(args.rounds, args.repetitions, args.reference_tokens) <= 0 or args.warmup < 0:
        parser.error("rounds, repetitions and reference-tokens must be positive; warmup must be nonnegative")
    if not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error("warmup-ms must be finite and nonnegative")
    if not 0 < args.cooldown_temperature < args.max_temperature < 110:
        parser.error("require 0 < cooldown-temperature < max-temperature < 110")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix")
    return args


def gpu_state():
    state = {}
    clock_path = Path("/sys/class/devfreq/gpu-gpc-0")
    for name in ("min_freq", "max_freq", "cur_freq"):
        try:
            state[name + "_hz"] = int((clock_path / name).read_text().strip())
        except (OSError, ValueError):
            pass
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=temperature.gpu,clocks.current.graphics,clocks_event_reasons.sw_thermal_slowdown", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip().splitlines()[0].split(",")
        state.update(
            temperature_c=float(value[0]),
            graphics_clock_mhz=float(value[1]),
            sw_thermal_slowdown=value[2].strip(),
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return state


def wait_for_cooling(args):
    deadline = time.monotonic() + 240
    announced = False
    while True:
        state = gpu_state()
        temperature = state.get("temperature_c")
        thermal = state.get("sw_thermal_slowdown") == "Active"
        if temperature is None:
            if thermal:
                raise RuntimeError("GPU thermal slowdown is active")
            return state
        if temperature <= args.cooldown_temperature and not thermal:
            return state
        if not announced:
            print(f"Cooling outside timing: {temperature:.1f} C; target <= {args.cooldown_temperature:g} C", flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU did not cool to the requested temperature within 240 seconds")
        time.sleep(2)


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    fields = ["tokens", "experts", "hidden", "intermediate", "top_k", "int8_median_ms", "float_eager_median_ms", "int8_effective_tops", "float_eager_effective_tflops", "speedup_over_eager_float", "int8_round_medians_ms", "float_eager_round_medians_ms"]
    csv_path = output.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in report["cases"]:
            row = {name: case[name] for name in fields[:5]}
            int8, baseline = case["results"]["int8_moe_ffn"], case["results"]["float_eager_experts"]
            row.update(
                int8_median_ms=int8["median_ms"], float_eager_median_ms=baseline["median_ms"],
                int8_effective_tops=int8["effective_tops"],
                float_eager_effective_tflops=baseline["effective_tflops"],
                speedup_over_eager_float=case["speedup_over_eager_float"],
                int8_round_medians_ms=json.dumps(int8["round_medians_ms"]),
                float_eager_round_medians_ms=json.dumps(baseline["round_medians_ms"]),
            )
            writer.writerow(row)
    temporary_csv.replace(csv_path)


def quantize_rows(x, torch):
    blocks = x.float().reshape(x.shape[0], -1, 256)
    scales = blocks.abs().amax(-1).clamp_min(1e-12) / 127
    q = (blocks / scales.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(x.shape), scales


def prepare_weight(weight, torch):
    e, n, k = weight.shape
    q, scales = quantize_rows(weight.reshape(e * n, k), torch)
    return q.reshape(e, n, k).transpose(-2, -1), scales.reshape(e, n, k // 256).transpose(-2, -1).contiguous()


def cpu_reference(x, ids, scores, w1, w2, s1, s2, torch, *, scale_x=None, output_dtype=None):
    """Exact CPU INT32 group dots, evaluated independently per top-k slot."""
    dtype = output_dtype or (torch.bfloat16 if x.dtype == torch.int8 else x.dtype)
    scale_x = scale_x.cpu() if scale_x is not None else None
    x, ids, scores, w1, w2, s1, s2 = (t.cpu() for t in (x, ids, scores, w1, w2, s1, s2))

    def project(a, sa, b, sb):
        out = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
        for group, start in enumerate(range(0, a.shape[1], 256)):
            dot = a[:, start : start + 256].int() @ b[start : start + 256].int()
            scale = sa[:, group, None] * sb[group, None, :]
            out = (out.double() + dot.double() * scale.double()).float()
        return out.to(dtype).float()

    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        if scale_x is None:
            qa, sa = quantize_rows(x[token : token + 1], torch)
        else:
            qa, sa = x[token : token + 1], scale_x[token : token + 1]
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = project(qa, sa, w1[expert], s1[expert]).chunk(2, -1)
            hidden = torch.nn.functional.silu(gate) * up
            qhidden, shidden = quantize_rows(hidden, torch)
            value = project(qhidden, shidden, w2[expert], s2[expert])
            output[token] += value[0] * scores[token, slot]
    return output.to(dtype)


def bf16_expert_reference(x, w1, w2, ids, scores, torch):
    """Full ragged MoE FFN using ordinary PyTorch Linear-format weights."""
    output = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        gate, up = torch.nn.functional.linear(x[token], w1[expert]).chunk(2, -1)
        hidden = torch.nn.functional.silu(gate) * up
        value = torch.nn.functional.linear(hidden, w2[expert])
        output.index_add_(0, token, value.float() * scores[token, slot, None])
    return output.to(x.dtype)


def summarize(samples, operations, integer):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("all latency samples must be finite and positive")
    median = statistics.median(samples)
    return dict(
        round_medians_ms=samples, median_ms=median, min_ms=min(samples), max_ms=max(samples),
        **{"effective_tops" if integer else "effective_tflops": operations / (median * 1e9)},
    )


def warmup(function, args, torch):
    """Warm one full FFN path without accumulating queued GPU allocations."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    calls = 0
    while calls < args.warmup or (time.perf_counter() - started) * 1000 < args.warmup_ms:
        function()
        torch.cuda.synchronize()
        calls += 1
    return dict(calls=calls, elapsed_ms=(time.perf_counter() - started) * 1000)


def source_metadata(package):
    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *arguments], text=True,
                stderr=subprocess.DEVNULL, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    native = package / "kernels" / "_native"
    return dict(
        git_commit=git("rev-parse", "HEAD"),
        git_status_porcelain=git("status", "--porcelain"),
        native_file_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(native.glob("*")) if path.is_file() and path.suffix in (".so", ".json")},
    )


def main():
    args = parse_args()
    import torch
    import triton
    import grain_gemm

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    e, h, i, r = args.experts, args.hidden, args.intermediate, args.top_k
    w1 = (torch.randn((e, 2 * i, h), device="cuda", dtype=dtype) / h**0.5).contiguous()
    w2 = (torch.randn((e, h, i), device="cuda", dtype=dtype) / i**0.5).contiguous()
    q1, s1 = prepare_weight(w1, torch)
    q2, s2 = prepare_weight(w2, torch)
    package = Path(grain_gemm.__file__).resolve().parent
    report = dict(
        complete=False, expected_tokens=args.tokens, seed=args.seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
        dtype=args.dtype, group_size=256, activation="swiglu", backend=args.backend,
        experts=e, hidden=h, intermediate=i, top_k=r,
        rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup,
        warmup_ms=args.warmup_ms,
        software=dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda, python=platform.python_version()),
        timing="Eager full-call CUDA events; rotating implementation order; median of round medians. Includes host dispatch gaps; CUDA Graphs are not used.",
        scope="Starts from prequantized INT8 activation and scale; includes token routing/packing, both expert GEMMs, SwiGLU, and FP32 weighted combination. Excludes weight quantization, input generation, compilation, warmup, correctness checks, and cooling.",
        floating_point_baseline="Readable per-expert PyTorch loop with the same BF16/FP16 dtype as hidden_states, dynamic torch.where routing, two Linear operations, SwiGLU and FP32 index_add. Not a tuned fused MoE backend.",
        throughput="6*T*top_k*H*I divided by full FFN latency; effective INT8 TOPS/floating-point TFLOPS exclude non-GEMM operation counts.",
        cpu_correctness="Independent per-token/per-slot CPU INT32 group dots, FP32 group scale products, GEMM output rounding, FP32 SwiGLU and re-quantization.",
        source=source_metadata(package),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package.rglob("*")) if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".json")},
        cases=[],
    )
    before = gpu_state()
    report["gpu_state_before"] = before
    if before.get("min_freq_hz") == before.get("max_freq_hz") == 1197000000:
        report["frequency_note"] = "GPU clock reduced to 1.2 GHz; actual supported locked setting is 1197 MHz."
    else:
        report["frequency_note"] = "No GPU frequency change is performed by this benchmark; see recorded frequency snapshots."
    save(report, args.output)

    with torch.inference_mode():
        for tokens in args.tokens:
            torch.manual_seed(args.seed + tokens)
            x = torch.randn((tokens, h), device="cuda", dtype=dtype)
            # Nonuniform counts arise naturally; no equal-M padding is supplied.
            ids = torch.randn((tokens, e), device="cuda").topk(r, dim=-1).indices.contiguous()
            scores = torch.rand((tokens, r), device="cuda")
            scores = (scores / scores.sum(-1, keepdim=True)).contiguous()

            qx, sx = quantize_rows(x, torch)

            def run_int8():
                return grain_gemm.moe_ffn(qx, q1, q2, scores, ids, scale_x=sx, scale_gate_up=s1, scale_down=s2, output_dtype=dtype, backend=args.backend)

            def run_bf16():
                return bf16_expert_reference(x, w1, w2, ids, scores, torch)

            actual, baseline = run_int8(), run_bf16()
            count = min(tokens, args.reference_tokens)
            expected = cpu_reference(x[:count], ids[:count], scores[:count], q1, q2, s1, s2, torch)
            torch.testing.assert_close(actual[:count].cpu(), expected, rtol=0.015, atol=0.025)
            quant_error = (actual.float() - baseline.float())
            check = dict(
                independent_cpu_tokens=count, independent_cpu_routes=count * r,
                int8_matches_cpu_reference=True,
                int8_vs_unquantized_float_max_abs=quant_error.abs().max().item(),
                int8_vs_unquantized_float_rmse=quant_error.square().mean().sqrt().item(),
                comparison_is_quantization_error_not_exact_equality=True,
            )
            torch.cuda.synchronize()
            state_before = wait_for_cooling(args)
            functions = dict(int8_moe_ffn=run_int8, float_eager_experts=run_bf16)
            # Warm after cooling so a lengthy idle period cannot undo warmup.
            # No subprocess or telemetry query intervenes before timing begins.
            warmup_results = {name: warmup(function, args, torch) for name, function in functions.items()}
            samples = {name: [] for name in functions}
            for round_index in range(args.rounds):
                names = list(functions)
                if round_index % 2:
                    names.reverse()
                for name in names:
                    repeats = []
                    for _ in range(args.repetitions):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        functions[name]()
                        end.record()
                        end.synchronize()
                        repeats.append(start.elapsed_time(end))
                    samples[name].append(statistics.median(repeats))
                state = gpu_state()
                if state.get("sw_thermal_slowdown") == "Active" or state.get("temperature_c", 0) >= args.max_temperature:
                    raise RuntimeError(f"Discarding thermally limited case T={tokens}: {state}")
            state_after = gpu_state()
            operations = 6 * tokens * r * h * i
            results = {name: summarize(values, operations, name == "int8_moe_ffn") for name, values in samples.items()}
            speedup = results["float_eager_experts"]["median_ms"] / results["int8_moe_ffn"]["median_ms"]
            case = dict(
                tokens=tokens, experts=e, hidden=h, intermediate=i, top_k=r,
                route_counts=torch.bincount(ids.flatten(), minlength=e).cpu().tolist(),
                checks=check, results=results, speedup_over_eager_float=speedup,
                warmup=warmup_results,
                gpu_state_before=state_before, gpu_state_after=state_after,
            )
            report["cases"].append(case)
            report["complete"] = len(report["cases"]) == len(args.tokens)
            report["gpu_state_after"] = state_after
            save(report, args.output)
            print(f"T={tokens:5d}: INT8 {results['int8_moe_ffn']['median_ms']:.3f} ms, eager {args.dtype} {results['float_eager_experts']['median_ms']:.3f} ms, ratio {speedup:.2f}x", flush=True)


if __name__ == "__main__":
    main()
