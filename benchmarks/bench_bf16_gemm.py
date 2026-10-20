"""Measure standalone BF16 square GEMM throughput, without INT8 or MoE work.

Example: python benchmarks/bench_bf16_gemm.py --sizes 1024 2048 4096 8192
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics

from bench_moe_ffn import ROOT, gpu_state, wait_for_cooling, warmup


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--max-temperature", type=float, default=85)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/thor/bf16_gemm_1197mhz.json")
    args = parser.parse_args()
    if min(*args.sizes, args.rounds, args.rep_ms) <= 0 or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes/rounds/rep-ms must be positive; sizes must be distinct")
    if args.warmup < 0 or not 0 <= args.warmup_ms < float("inf"):
        parser.error("warmup counts and milliseconds must be finite and nonnegative")
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
    temporary_csv = output.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as stream:
        fields = ["m", "n", "k", "method", "layout", "api", "median_us", "tflops",
                  "round_medians_ms"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case in report["cases"]:
            for name, result in case["results"].items():
                writer.writerow(dict(
                    m=case["size"], n=case["size"], k=case["size"], method=name,
                    layout=result["layout"], api=result["api"],
                    median_us=result["median_ms"] * 1000, tflops=result["tflops"],
                    round_medians_ms=json.dumps(result["round_medians_ms"]),
                ))
    temporary_csv.replace(output.with_suffix(".csv"))


def main():
    args = parse_args()
    import torch
    import triton
    from triton.testing import do_bench_cudagraph

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    properties = torch.cuda.get_device_properties()
    scripts = [Path(__file__), Path(__file__).with_name("bench_moe_ffn.py"),
               Path(inspect.getfile(do_bench_cudagraph))]
    state = gpu_state()
    report = dict(
        complete=False, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu=properties.name, compute_capability=list(torch.cuda.get_device_capability()),
        multiprocessor_count=properties.multi_processor_count, dtype="bfloat16",
        expected_sizes=args.sizes, seed=args.seed,
        software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                      triton=triton.__version__, python=platform.python_version()),
        blas_preference=str(torch.backends.cuda.preferred_blas_library()),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        timing=dict(method="triton.testing.do_bench_cudagraph", rounds=args.rounds,
                    rep_ms=args.rep_ms, warmup_ms=args.warmup_ms,
                    statistic="median of round medians; methods rotate order each round",
                    scope="BF16 GEMM only; excludes host dispatch, input generation, layout preparation, correctness checks and warmup",
                    rep_note="Installed Triton estimates calls per captured batch and replays the batch ten times",
                    cache_policy="repeated inputs and weights, no explicit L2 flush"),
        throughput="2*M*N*K/(median_seconds*1e12), dense FLOPs; one FMA counts as two",
        script_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts},
        gpu_state_before=state, cases=[],
    )
    report["frequency_note"] = (
        "GPU frequency reduced to 1.2 GHz; actual supported locked setting is 1197 MHz."
        if state.get("min_freq_hz") == state.get("max_freq_hz") == 1197000000
        else "The benchmark does not change clocks; see the recorded frequency snapshots."
    )
    save(report, args.output)
    with torch.inference_mode():
        for size in args.sizes:
            torch.manual_seed(args.seed + size)
            a = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
            b_nt = weight.t()
            b_nn = b_nt.contiguous()
            c_nt = torch.empty_like(a)
            c_nn = torch.empty_like(a)
            functions = dict(
                nt_default=lambda: torch.mm(a, b_nt),
                nt_out=lambda: torch.mm(a, b_nt, out=c_nt),
                nn_default=lambda: torch.mm(a, b_nn),
                nn_out=lambda: torch.mm(a, b_nn, out=c_nn),
            )
            indices = torch.linspace(0, size - 1, 8, device="cuda").long().unique()
            sample_a = a[indices].cpu().double()
            sample_weight = weight[indices].cpu().double()
            expected = (sample_a @ sample_weight.t()).bfloat16()
            checks = {}
            for name, function in functions.items():
                actual = function()
                if not torch.isfinite(actual).all().item():
                    raise AssertionError(f"Nonfinite BF16 GEMM result: {size}, {name}")
                sample = actual[indices[:, None], indices[None, :]].cpu()
                torch.testing.assert_close(sample, expected, rtol=0.016, atol=0.005)
                delta = sample.float() - expected.float()
                checks[name] = dict(
                    independent_cpu_fp64_samples=sample.numel(),
                    max_abs=delta.abs().max().item(),
                    relative_l2=(delta.norm() / expected.float().norm().clamp_min(1e-12)).item(),
                    all_outputs_finite=True,
                )
            torch.cuda.synchronize()
            before = wait_for_cooling(args)
            warmup_results = {name: warmup(function, args, torch)
                              for name, function in functions.items()}
            samples = {name: [] for name in functions}
            orders = []
            for round_index in range(args.rounds):
                names = list(functions)
                rotate = round_index % len(names)
                names = names[rotate:] + names[:rotate]
                orders.append(names)
                for name in names:
                    samples[name].append(do_bench_cudagraph(
                        functions[name], rep=args.rep_ms, return_mode="median"))
                state = gpu_state()
                if state.get("sw_thermal_slowdown") == "Active" or state.get("temperature_c", 0) >= args.max_temperature:
                    raise RuntimeError(f"Thermally limited measurement at {size}: {state}")
            after = gpu_state()
            results = {}
            for name, values in samples.items():
                median = statistics.median(values)
                results[name] = dict(
                    layout="NT, B column-major" if name.startswith("nt") else "NN, B row-major",
                    api="torch.mm(out=C)" if name.endswith("out") else "torch.mm",
                    round_medians_ms=values, median_ms=median,
                    min_ms=min(values), max_ms=max(values),
                    tflops=2 * size**3 / (median * 1e9),
                )
            case = dict(size=size, results=results, checks=checks, warmup=warmup_results,
                        measurement_orders=orders, a_stride=list(a.stride()),
                        b_nt_stride=list(b_nt.stride()), b_nn_stride=list(b_nn.stride()),
                        gpu_state_before=before, gpu_state_after=after)
            report["cases"].append(case)
            report["complete"] = len(report["cases"]) == len(args.sizes)
            report["gpu_state_after"] = after
            report["best_observed"] = max(
                (dict(size=c["size"], method=name, median_ms=result["median_ms"], tflops=result["tflops"])
                 for c in report["cases"] for name, result in c["results"].items()),
                key=lambda value: value["tflops"],
            )
            save(report, args.output)
            best = max(results, key=lambda name: results[name]["tflops"])
            print(f"{size}: NT {results['nt_default']['tflops']:.2f}, NT out "
                  f"{results['nt_out']['tflops']:.2f}, NN {results['nn_default']['tflops']:.2f}, "
                  f"NN out {results['nn_out']['tflops']:.2f} TFLOPS; best {best}; "
                  f"{before.get('temperature_c')} -> {after.get('temperature_c')} C", flush=True)


if __name__ == "__main__":
    main()
