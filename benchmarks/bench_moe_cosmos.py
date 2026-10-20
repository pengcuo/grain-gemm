"""Compare the public INT8 FFN with the original Cosmos grouped-MM BF16 FFN.

Example: PYTHONPATH=src python benchmarks/bench_moe_cosmos.py \
    --cosmos-dir /path/to/cosmos-framework

All methods include expert counts, dispatch, both projections and combination.
Cosmos is measured eagerly and with its benchmark's default torch.compile
configuration. Router/top-k selection and weight preparation are excluded.
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
import time

from bench_moe_ffn import (
    ROOT, cpu_reference, gpu_state, prepare_weight, source_metadata, summarize,
    wait_for_cooling, warmup,
)
from cosmos_moe_baseline import load_cosmos_experts, make_cosmos_forward


METHODS = ("int8_auto", "cosmos_bf16_eager", "cosmos_bf16_compiled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-tokens", type=int, default=2)
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--max-temperature", type=float, default=85)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/thor/moe_ffn_cosmos.json")
    args = parser.parse_args()
    if not args.tokens or min(args.tokens) <= 0 or len(set(args.tokens)) != len(args.tokens):
        parser.error("tokens must be distinct positive integers")
    if min(args.experts, args.hidden, args.intermediate, args.top_k) <= 0:
        parser.error("expert count, dimensions and top-k must be positive")
    if args.hidden % 256 or args.intermediate % 256 or args.top_k > args.experts:
        parser.error("H/I must be multiples of 256 and top-k must not exceed E")
    if min(args.rounds, args.repetitions, args.reference_tokens) <= 0 or args.warmup < 0:
        parser.error("rounds/repetitions/reference-tokens must be positive; warmup nonnegative")
    if not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error("warmup-ms must be finite and nonnegative")
    if not 0 < args.cooldown_temperature < args.max_temperature < 110:
        parser.error("require 0 < cooldown-temperature < max-temperature < 110")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix")
    return args


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    fields = ["tokens", "experts", "hidden", "intermediate", "top_k"]
    for method in METHODS:
        fields += [method + "_ms", method + "_effective_tops_or_tflops",
                   method + "_round_medians_ms"]
    fields += ["eager_bf16_over_int8", "compiled_bf16_over_int8"]
    temporary_csv = output.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in report["cases"]:
            row = {name: case[name] for name in fields[:5]}
            for method in METHODS:
                result = case["results"][method]
                row[method + "_ms"] = result["median_ms"]
                row[method + "_effective_tops_or_tflops"] = result.get(
                    "effective_tops", result.get("effective_tflops"))
                row[method + "_round_medians_ms"] = json.dumps(result["round_medians_ms"])
            row.update(case["ratios"])
            writer.writerow(row)
    temporary_csv.replace(output.with_suffix(".csv"))


def bf16_cpu_reference(x, ids, scores, w1, w2, torch, *, fused_activation=False):
    """Independent per-token/slot reference; scores multiply before down.

    Each projection rounds to BF16. The eager reference also rounds every
    pointwise intermediate; the fused reference rounds the SwiGLU/score product
    once, as an Inductor pointwise fusion may do. Combination is accumulated in
    FP32; the actual BF16 scatter may differ by a few rounding units.
    """
    x, ids, scores, w1, w2 = (tensor.cpu() for tensor in (x, ids, scores, w1, w2))
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = (x[token].float() @ w1[expert].float().t()).bfloat16()
            gate, up = gate_up.chunk(2, -1)
            if fused_activation:
                hidden = (torch.nn.functional.silu(gate.float()) * up.float()
                          * scores[token, slot].float()).bfloat16()
            else:
                hidden = torch.nn.functional.silu(gate).bfloat16()
                hidden = (hidden * up).bfloat16()
                hidden = (hidden * scores[token, slot].bfloat16()).bfloat16()
            output[token] += (hidden.float() @ w2[expert].float().t()).bfloat16().float()
    return output.bfloat16()


def difference(actual, expected, torch):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return dict(
        max_abs=delta.abs().max().item(),
        rmse=delta.square().mean().sqrt().item(),
        relative_l2=(delta.norm() / expected.norm().clamp_min(1e-12)).item(),
    )


def main():
    args = parse_args()
    import torch
    import triton
    import grain_gemm

    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    e, h, i, r = args.experts, args.hidden, args.intermediate, args.top_k
    dtype = torch.bfloat16
    w1 = (torch.randn((e, 2 * i, h), device="cuda", dtype=dtype) / h**0.5).contiguous()
    w2 = (torch.randn((e, h, i), device="cuda", dtype=dtype) / i**0.5).contiguous()
    q1, s1 = prepare_weight(w1, torch)
    q2, s2 = prepare_weight(w2, torch)
    module, cosmos_metadata = load_cosmos_experts(
        args.cosmos_dir, e, h, i, r, "cuda", dtype, w1, w2,
    )
    eager = make_cosmos_forward(module)
    # Match Cosmos moe_bench.py's default, without extra autotuning or Graphs.
    compiled = torch.compile(eager, fullgraph=True, dynamic=True)
    package = Path(grain_gemm.__file__).resolve().parent
    scripts = (Path(__file__), Path(__file__).with_name("cosmos_moe_baseline.py"),
               Path(__file__).with_name("bench_moe_ffn.py"))
    before = gpu_state()
    report = dict(
        complete=False, seed=args.seed, expected_tokens=args.tokens,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
        dtype="bfloat16", experts=e, hidden=h, intermediate=i, top_k=r, group_size=256,
        activation="swiglu", int8_backend="auto", rounds=args.rounds,
        repetitions=args.repetitions, warmup=args.warmup, warmup_ms=args.warmup_ms,
        software=dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                      python=platform.python_version()),
        cosmos=cosmos_metadata,
        compilation=dict(fullgraph=True, dynamic=True, backend="inductor", mode="default",
                         boundary="counts plus original Cosmos expert forward"),
        timing="Full-call CUDA events including host dispatch gaps; rotating three-method order; median of round medians; no CUDA Graph capture.",
        scope="Includes expert histogram, sort/pad/gather dispatch, two projections, activation and combine; INT8 includes dynamic activation quantization. Excludes router/top-k, weight preparation, compilation, warmup, correctness and cooling.",
        score_policy="FP32 random scores normalized once, rounded to BF16, then supplied to INT8 as FP32 with identical numeric values. No extra INT8 normalization.",
        cosmos_score_input_dtype="BF16, prepared before timing as in the original Cosmos expert interface",
        numerical_difference="Cosmos multiplies routing score before down and uses BF16 scatter_add. INT8 multiplies after down and reduces in FP32. Their rounding and quantization contracts differ; equality between methods is not asserted.",
        throughput="6*T*top_k*H*I divided by complete FFN latency; non-GEMM operation counts excluded. INT8 in effective TOPS, BF16 in effective TFLOPS.",
        source=source_metadata(package),
        source_sha256={str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(package.rglob("*")) if path.is_file()
                       and path.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".json")},
        benchmark_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts},
        gpu_state_before=before, cases=[],
    )
    report["frequency_note"] = (
        "GPU clock reduced to 1.2 GHz; actual supported locked setting is 1197 MHz."
        if before.get("min_freq_hz") == before.get("max_freq_hz") == 1197000000
        else "The benchmark does not change clocks; see the recorded frequency snapshots."
    )
    save(report, args.output)

    with torch.inference_mode():
        for tokens in args.tokens:
            torch.manual_seed(args.seed + tokens)
            x = torch.randn((tokens, h), device="cuda", dtype=dtype)
            ids = torch.randn((tokens, e), device="cuda").topk(r, dim=-1).indices.contiguous()
            scores = torch.rand((tokens, r), device="cuda")
            scores_bf16 = (scores / scores.sum(-1, keepdim=True)).bfloat16().contiguous()
            scores = scores_bf16.float()
            functions = dict(
                int8_auto=lambda: grain_gemm.moe_ffn(
                    x, q1, q2, scores, ids, scale_gate_up=s1, scale_down=s2),
                cosmos_bf16_eager=lambda: eager(x, ids, scores_bf16),
                cosmos_bf16_compiled=lambda: compiled(x, ids, scores_bf16),
            )
            outputs = {}
            compile_seconds = {}
            for name, function in functions.items():
                started = time.perf_counter()
                try:
                    outputs[name] = function()
                    torch.cuda.synchronize()
                except Exception as error:
                    report["failure"] = dict(tokens=tokens, method=name,
                                             error=f"{type(error).__name__}: {error}")
                    save(report, args.output)
                    raise
                compile_seconds[name] = time.perf_counter() - started
                if not torch.isfinite(outputs[name]).all().item():
                    raise AssertionError(f"Nonfinite output from {name}")
            count = min(tokens, args.reference_tokens)
            int8_expected = cpu_reference(
                x[:count], ids[:count], scores[:count], q1, q2, s1, s2, torch)
            torch.testing.assert_close(outputs["int8_auto"][:count].cpu(), int8_expected,
                                       rtol=0.015, atol=0.025)
            checks = dict(independent_cpu_tokens=count, independent_cpu_routes=count * r,
                          int8_matches_int32_reference=True, all_outputs_finite=True)
            for name, fused in (("cosmos_bf16_eager", False), ("cosmos_bf16_compiled", True)):
                expected = bf16_cpu_reference(
                    x[:count], ids[:count], scores[:count], w1, w2, torch,
                    fused_activation=fused,
                )
                torch.testing.assert_close(outputs[name][:count].cpu(), expected,
                                           rtol=0.025, atol=0.008)
                checks[name + "_vs_cpu"] = difference(outputs[name][:count].cpu(), expected, torch)
                checks["int8_vs_" + name] = difference(outputs["int8_auto"], outputs[name], torch)
            checks["cosmos_compiled_vs_eager"] = difference(
                outputs["cosmos_bf16_compiled"], outputs["cosmos_bf16_eager"], torch)
            torch.cuda.synchronize()
            state_before = wait_for_cooling(args)
            warmup_results = {name: warmup(function, args, torch)
                              for name, function in functions.items()}
            samples = {name: [] for name in functions}
            for round_index in range(args.rounds):
                names = list(functions)
                rotate = round_index % len(names)
                for name in names[rotate:] + names[:rotate]:
                    values = []
                    for _ in range(args.repetitions):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        functions[name]()
                        end.record()
                        end.synchronize()
                        values.append(start.elapsed_time(end))
                    samples[name].append(statistics.median(values))
                state = gpu_state()
                if state.get("sw_thermal_slowdown") == "Active" or state.get("temperature_c", 0) >= args.max_temperature:
                    raise RuntimeError(f"Discarding thermally limited case T={tokens}: {state}")
            state_after = gpu_state()
            operations = 6 * tokens * r * h * i
            results = {name: summarize(values, operations, name == "int8_auto")
                       for name, values in samples.items()}
            int8_ms = results["int8_auto"]["median_ms"]
            ratios = dict(eager_bf16_over_int8=results["cosmos_bf16_eager"]["median_ms"] / int8_ms,
                          compiled_bf16_over_int8=results["cosmos_bf16_compiled"]["median_ms"] / int8_ms)
            case = dict(tokens=tokens, experts=e, hidden=h, intermediate=i, top_k=r,
                        route_counts=torch.bincount(ids.flatten(), minlength=e).cpu().tolist(),
                        checks=checks, first_call_seconds=compile_seconds, warmup=warmup_results,
                        results=results, ratios=ratios, gpu_state_before=state_before,
                        gpu_state_after=state_after)
            report["cases"].append(case)
            report["complete"] = len(report["cases"]) == len(args.tokens)
            report["gpu_state_after"] = state_after
            save(report, args.output)
            print(f"T={tokens:5d}: INT8 {int8_ms:.3f} ms; Cosmos BF16 eager "
                  f"{results['cosmos_bf16_eager']['median_ms']:.3f} ms; compiled "
                  f"{results['cosmos_bf16_compiled']['median_ms']:.3f} ms; compiled/INT8 "
                  f"{ratios['compiled_bf16_over_int8']:.3f}x", flush=True)


if __name__ == "__main__":
    main()
