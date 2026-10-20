"""Measure the upstream vLLM BF16 expert FFN used by vLLM-Omni.

Both methods include pre-routed dispatch, gate/up, SwiGLU, down and combine.
This is an expert-core comparison, not an Omni serving-engine benchmark.
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

from bench_moe_ffn import ROOT, gpu_state, source_metadata, summarize, wait_for_cooling, warmup
from bench_moe_cosmos import bf16_cpu_reference, difference
from cosmos_moe_baseline import load_cosmos_experts, make_cosmos_forward
from vllm_moe_baseline import load_vllm_experts


METHODS = ("vllm_bf16", "cosmos_bf16_compiled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--omni-dir", type=Path, required=True)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048, 3072, 4096])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--reference-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--max-temperature", type=float, default=85)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/thor/moe_ffn_vllm.json")
    args = parser.parse_args()
    if min(*args.tokens, args.experts, args.hidden, args.intermediate, args.top_k,
           args.rounds, args.repetitions, args.reference_tokens) <= 0:
        parser.error("dimensions and measurement counts must be positive")
    if len(set(args.tokens)) != len(args.tokens) or args.top_k > args.experts:
        parser.error("tokens must be distinct and top-k must not exceed the expert count")
    if args.warmup < 0 or not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error("warmup must be finite and nonnegative")
    if not 0 < args.cooldown_temperature < args.max_temperature < 110:
        parser.error("require 0 < cooldown-temperature < max-temperature < 110")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix")
    return args


def vllm_cpu_reference(x, ids, scores, w1, w2, torch):
    """Independent token/route reference with the upstream BF16 rounding points.

    CPU FP64 dots cover full H/I. Native silu_and_mul rounds SiLU to BF16
    before multiplying up. FC2 multiplies its FP32 accumulator by the score
    before rounding to BF16; the top-k reduction accumulates in FP32.
    """
    x, ids, scores, w1, w2 = (value.cpu() for value in (x, ids, scores, w1, w2))
    output = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = (x[token].double() @ w1[expert].double().t()).bfloat16()
            gate, up = gate_up.chunk(2, -1)
            silu = torch.nn.functional.silu(gate.float()).bfloat16()
            hidden = (silu.float() * up.float()).bfloat16()
            down = hidden.double() @ w2[expert].double().t()
            weighted = (down * scores[token, slot].double()).bfloat16()
            output[token] += weighted.float()
    return output.bfloat16()


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    fields = ["tokens", "experts", "hidden", "intermediate", "top_k"]
    for method in METHODS:
        fields += [method + "_ms", method + "_effective_tflops", method + "_round_medians_ms"]
    fields += ["cosmos_over_vllm"]
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            row = {key: case[key] for key in fields[:5]}
            for method in METHODS:
                result = case["results"][method]
                row[method + "_ms"] = result["median_ms"]
                row[method + "_effective_tflops"] = result["effective_tflops"]
                row[method + "_round_medians_ms"] = json.dumps(result["round_medians_ms"])
            row["cosmos_over_vllm"] = case["cosmos_over_vllm"]
            writer.writerow(row)
    temporary.replace(output.with_suffix(".csv"))


def omni_metadata(root):
    root = root.expanduser().resolve()
    paths = [Path("vllm_omni/diffusion/layers/fused_moe.py"),
             Path("vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py")]
    return dict(
        source_dir=str(root),
        git_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        scope="Omni adapter provenance only; executes its upstream Triton functional expert core, without engine, model router, distributed runner or forward-context hooks",
    )


def main():
    args = parse_args()
    import torch
    import triton
    import grain_gemm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    vllm_forward, vllm_metadata = load_vllm_experts(args.vllm_dir)
    torch.manual_seed(args.seed)
    e, h, i, r = args.experts, args.hidden, args.intermediate, args.top_k
    w1 = (torch.randn((e, 2 * i, h), device="cuda", dtype=torch.bfloat16) / h**0.5).contiguous()
    w2 = (torch.randn((e, h, i), device="cuda", dtype=torch.bfloat16) / i**0.5).contiguous()
    module, cosmos_metadata = load_cosmos_experts(
        args.cosmos_dir, e, h, i, r, "cuda", torch.bfloat16, w1, w2)
    cosmos_forward = torch.compile(make_cosmos_forward(module), fullgraph=True, dynamic=True)
    scripts = [Path(__file__), *(Path(__file__).with_name(name) for name in
               ["vllm_moe_baseline.py", "vllm_moe_native.py", "cosmos_moe_baseline.py",
                "bench_moe_ffn.py", "bench_moe_cosmos.py"])]
    before = gpu_state()
    report = dict(
        complete=False, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        expected_tokens=args.tokens, experts=e, hidden=h, intermediate=i, top_k=r,
        seed=args.seed, dtype="bfloat16", activation="swiglu",
        gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
        software=dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                      python=platform.python_version()),
        rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup, warmup_ms=args.warmup_ms,
        timing="Full-call CUDA events including host dispatch gaps; rotating method order; median of round medians; no CUDA Graph capture",
        scope="Pre-routed complete expert FFN: dispatch/alignment, gate/up, SwiGLU, down and combination. Excludes router/top-k, weight preparation, compilation, warmup, correctness and cooling. No Omni serving-engine overhead.",
        throughput="6*T*top_k*H*I / (latency_seconds*1e12), effective dense TFLOPS; includes gate/up plus down, excludes padding and non-GEMM operation counts",
        score_policy="Normalize FP32 scores once and round to BF16 outside timing. Cosmos receives BF16; vLLM receives FP32 with exactly the same numeric values.",
        numerical_difference="Cosmos compiled fuses pointwise operations and multiplies score before down; vLLM native SiLU rounds before multiply, and score multiplies FC2 accumulator before BF16 conversion, then FP32 top-k sum. Same real-valued FFN, different rounding.",
        correctness=dict(vllm_reference="Independent CPU FP64 full-H/I dot products with explicit BF16 rounding points",
                         cosmos_reference="Independent CPU FP32 full-H/I dot products with fused-pointwise BF16 reference",
                         rtol=0.025, atol=0.008, reference_tokens=args.reference_tokens),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cosmos_compilation=dict(fullgraph=True, dynamic=True, backend="inductor", mode="default"),
        vllm=vllm_metadata, omni=omni_metadata(args.omni_dir), cosmos=cosmos_metadata,
        source=source_metadata(Path(grain_gemm.__file__).resolve().parent),
        benchmark_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts},
        w1_shape=list(w1.shape), w1_stride=list(w1.stride()),
        w2_shape=list(w2.shape), w2_stride=list(w2.stride()),
        gpu_state_before=before, cases=[],
    )
    report["frequency_note"] = (
        "GPU frequency reduced to 1.2 GHz; actual supported locked setting is 1197 MHz."
        if before.get("min_freq_hz") == before.get("max_freq_hz") == 1197000000
        else "Clocks unchanged by benchmark; see snapshots."
    )
    save(report, args.output)
    with torch.inference_mode():
        for tokens in args.tokens:
            torch.manual_seed(args.seed + tokens)
            x = torch.randn((tokens, h), device="cuda", dtype=torch.bfloat16)
            ids = torch.randn((tokens, e), device="cuda").topk(r, dim=-1).indices.contiguous()
            scores = torch.rand((tokens, r), device="cuda")
            scores_bf16 = (scores / scores.sum(-1, keepdim=True)).bfloat16().contiguous()
            scores = scores_bf16.float()
            vllm_config = vllm_forward.config_for(tokens, w1, w2, r)
            functions = dict(
                vllm_bf16=lambda: vllm_forward(x, w1, w2, scores, ids),
                cosmos_bf16_compiled=lambda: cosmos_forward(x, ids, scores_bf16),
            )
            outputs, first_call_seconds = {}, {}
            for name, function in functions.items():
                started = time.perf_counter()
                try:
                    outputs[name] = function()
                    torch.cuda.synchronize()
                    if not torch.isfinite(outputs[name]).all().item():
                        raise AssertionError(f"Nonfinite output from {name}")
                except Exception as error:
                    report["failure"] = dict(tokens=tokens, method=name, error=f"{type(error).__name__}: {error}")
                    save(report, args.output)
                    raise
                first_call_seconds[name] = time.perf_counter() - started
            count = min(tokens, args.reference_tokens)
            expected = vllm_cpu_reference(x[:count], ids[:count], scores[:count], w1, w2, torch)
            torch.testing.assert_close(outputs["vllm_bf16"][:count].cpu(), expected, rtol=0.025, atol=0.008)
            checks = dict(independent_cpu_tokens=count, independent_cpu_routes=count*r,
                          all_outputs_finite=True,
                          vllm_vs_cpu=difference(outputs["vllm_bf16"][:count].cpu(), expected, torch))
            expected = bf16_cpu_reference(x[:count], ids[:count], scores[:count], w1, w2, torch, fused_activation=True)
            torch.testing.assert_close(outputs["cosmos_bf16_compiled"][:count].cpu(), expected, rtol=0.025, atol=0.008)
            checks["cosmos_compiled_vs_cpu"] = difference(outputs["cosmos_bf16_compiled"][:count].cpu(), expected, torch)
            checks["vllm_vs_cosmos_compiled"] = difference(outputs["vllm_bf16"], outputs["cosmos_bf16_compiled"], torch)
            torch.cuda.synchronize()
            state_before = wait_for_cooling(args)
            warmup_results = {name: warmup(function, args, torch) for name, function in functions.items()}
            samples = {name: [] for name in functions}
            sample_values, orders, round_gpu_states = [], [], []
            for round_index in range(args.rounds):
                names = list(functions)
                rotate = round_index % len(names)
                names = names[rotate:] + names[:rotate]
                orders.append(names)
                current = {}
                for name in names:
                    values = []
                    for _ in range(args.repetitions):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        functions[name]()
                        end.record()
                        end.synchronize()
                        values.append(start.elapsed_time(end))
                    current[name] = values
                    samples[name].append(statistics.median(values))
                sample_values.append(current)
                state = gpu_state()
                round_gpu_states.append(state)
                if state.get("sw_thermal_slowdown") == "Active" or state.get("temperature_c", 0) >= args.max_temperature:
                    raise RuntimeError(f"Thermally limited case T={tokens}: {state}")
            after = gpu_state()
            results = {name: summarize(values, 6*tokens*r*h*i, False) for name, values in samples.items()}
            case = dict(tokens=tokens, experts=e, hidden=h, intermediate=i, top_k=r,
                        vllm_config=vllm_config,
                        route_counts=torch.bincount(ids.flatten(), minlength=e).cpu().tolist(),
                        checks=checks, first_call_seconds=first_call_seconds, warmup=warmup_results,
                        results=results, sample_values_ms=sample_values, measurement_orders=orders,
                        round_gpu_states=round_gpu_states, gpu_state_before=state_before, gpu_state_after=after,
                        cosmos_over_vllm=results["cosmos_bf16_compiled"]["median_ms"]/results["vllm_bf16"]["median_ms"])
            report["cases"].append(case)
            report["complete"] = len(report["cases"]) == len(args.tokens)
            report["gpu_state_after"] = after
            save(report, args.output)
            print(f"T={tokens}: vLLM {results['vllm_bf16']['median_ms']:.3f} ms / "
                  f"{results['vllm_bf16']['effective_tflops']:.2f} TFLOPS; Cosmos compiled "
                  f"{results['cosmos_bf16_compiled']['median_ms']:.3f} ms / "
                  f"{results['cosmos_bf16_compiled']['effective_tflops']:.2f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
