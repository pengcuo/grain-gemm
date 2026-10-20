"""Paired full-FFN comparison of old/new INT8 and original BF16 MoE baselines.

Example: PYTHONPATH=src python benchmarks/bench_moe_optimized.py \
    --before-package-dir /path/to/frozen/grain_gemm \
    --vllm-dir /path/to/vllm --omni-dir /path/to/vllm-omni \
    --cosmos-dir /path/to/cosmos-framework
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from bench_moe_ffn import (
    ROOT, cpu_reference, gpu_state as read_gpu_state, prepare_weight, source_metadata, summarize,
    wait_for_cooling, warmup,
)
from bench_moe_cosmos import bf16_cpu_reference, difference
from bench_moe_vllm import omni_metadata, vllm_cpu_reference
from cosmos_moe_baseline import load_cosmos_experts, make_cosmos_forward
from vllm_moe_baseline import load_vllm_experts


BEFORE_REVISION = "e4c3614f9e01dbb4d40bc349b372f53479f729c1"
METHODS = ("int8_before", "int8_auto", "vllm_bf16", "cosmos_bf16_compiled")
RATIOS = ("before_over_auto", "vllm_bf16_over_auto", "cosmos_bf16_over_auto", "cosmos_over_vllm")
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".h", ".hpp", ".json"}


def gpu_state():
    # Sample EMC before invoking NVML, whose host latency can let the dynamic
    # memory clock return to idle. This benchmark never changes either policy.
    state = {}
    for name in ("cur_freq", "min_freq", "max_freq"):
        try:
            state["emc_" + name + "_hz"] = int(
                (Path("/sys/class/devfreq/bwmgr") / name).read_text().strip())
        except (OSError, ValueError):
            pass
    state.update(read_gpu_state())
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "omni", "cosmos"):
        parser.add_argument(f"--{name}-dir", type=Path, required=True)
    parser.add_argument("--before-package-dir", type=Path, required=True,
                        help="Frozen e4c3614 package, required for the paired before measurement")
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048, 3072, 4096])
    for name, default in (("experts", 32), ("hidden", 1536), ("intermediate", 1024),
                          ("top-k", 4), ("rounds", 5), ("repetitions", 5),
                          ("reference-tokens", 2), ("seed", 2026)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--max-temperature", type=float, default=85)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/thor/moe_ffn_optimized.json")
    args = parser.parse_args()
    if min(*args.tokens, args.experts, args.hidden, args.intermediate, args.top_k,
           args.rounds, args.repetitions, args.reference_tokens) <= 0:
        parser.error("dimensions and measurement counts must be positive")
    if len(set(args.tokens)) != len(args.tokens) or args.top_k > args.experts:
        parser.error("tokens must be distinct and top-k must not exceed E")
    if args.hidden % 256 or args.intermediate % 256:
        parser.error("INT8 requires H and I to be multiples of 256")
    if args.warmup < 0 or not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error("warmup must be finite and nonnegative")
    if not 0 < args.cooldown_temperature < args.max_temperature < 110:
        parser.error("require 0 < cooldown-temperature < max-temperature < 110")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix")
    return args


def hash_package(package):
    package = Path(package).resolve()
    source = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "_native" not in path.relative_to(package).parts
    }
    native_dir = package / "kernels/_native"
    native = {
        str(path.relative_to(native_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(native_dir.rglob("*"))
        if path.is_file() and path.suffix in {".so", ".json"}
    }
    return dict(package_dir=str(package), source_sha256=source, native_file_sha256=native)


def load_before_package(directory):
    """Load a frozen package under its own name and verify its committed sources."""
    directory = directory.expanduser().resolve()
    if not (directory / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing paired baseline package: {directory}")
    metadata = hash_package(directory)
    prefix = "src/grain_gemm/"
    entries = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "-r", "--format=%(objectname) %(path)", BEFORE_REVISION, "--", "src/grain_gemm"],
        text=True,
    ).splitlines()
    verified = []
    for entry in entries:
        blob, tracked_path = entry.split(" ", 1)
        relative = tracked_path.removeprefix(prefix)
        if Path(relative).suffix not in SOURCE_SUFFIXES:
            continue
        original = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", blob])
        expected = hashlib.sha256(original).hexdigest()
        if metadata["source_sha256"].get(relative) != expected:
            raise ValueError(f"Frozen baseline differs from {BEFORE_REVISION}: {relative}")
        verified.append(relative)
    extras = set(metadata["source_sha256"]) - set(verified)
    if extras or not verified:
        raise ValueError(f"Unexpected or missing baseline sources: {sorted(extras)}")
    alias = "_grain_moe_before_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(alias, directory / "__init__.py",
                                                 submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    metadata.update(git_commit=BEFORE_REVISION, alias=alias,
                    verified_committed_source_files=len(verified),
                    native_note="Frozen locally built binaries and manifests; their actual hashes are recorded separately from committed sources")
    return module, metadata


def record_dispatch(package, function):
    """Observe an untimed real call, including any routing-driven fallback."""
    implementation = importlib.import_module(package.__name__ + ".moe")
    native = implementation._native_gemm
    calls = []

    def observe(a, weight, sa, sb, out, backend, **kwargs):
        config, counts_aware = 32, False
        if hasattr(implementation, "_native_config"):
            kernels = importlib.import_module(package.__name__ + ".kernels.cutlass_moe")
            if kernels.is_available(a.device):
                config = implementation._native_config(
                    a.shape[1], weight.shape[1], a.shape[2], experts=a.shape[0],
                )
                counts_aware = kernels.CONFIGS[config].get("counts_aware", False)
        calls.append(dict(backend=backend, a_shape=list(a.shape), weight_shape=list(weight.shape),
                          config_id=config, counts_aware=counts_aware,
                          device_counts_supplied=kwargs.get("counts") is not None))
        return native(a, weight, sa, sb, out, backend, **kwargs)

    with patch.object(implementation, "_native_gemm", observe):
        function()
    return dict(backend="cutlass" if calls else "triton", gemms=calls)


def verify_provenance(report, helpers):
    for name in ("before", "after"):
        expected = report[name]
        actual = hash_package(expected["package_dir"])
        for field in ("source_sha256", "native_file_sha256"):
            if actual[field] != expected[field]:
                raise RuntimeError(f"{name} {field} changed during measurement")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in helpers}
    if hashes != report["benchmark_sha256"]:
        raise RuntimeError("Benchmark or build helper changed during measurement")


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    fields = ["tokens", "experts", "hidden", "intermediate", "top_k"]
    for name in METHODS:
        unit = "effective_tops" if name.startswith("int8") else "effective_tflops"
        fields += [name + "_ms", name + "_" + unit, name + "_round_medians_ms", name + "_samples_ms"]
    fields += list(RATIOS)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            row = {key: case[key] for key in fields[:5]}
            for name, result in case["results"].items():
                unit = "effective_tops" if name.startswith("int8") else "effective_tflops"
                row.update({name + "_ms": result["median_ms"], name + "_" + unit: result[unit],
                            name + "_round_medians_ms": json.dumps(result["round_medians_ms"]),
                            name + "_samples_ms": json.dumps([values[name] for values in case["sample_values_ms"]])})
            row.update(case["ratios"])
            writer.writerow(row)
    temporary.replace(output.with_suffix(".csv"))


def main():
    args = parse_args()
    import torch
    import triton
    import grain_gemm

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("This comparison requires Thor SM110")
    torch.set_num_threads(4)
    before_package, before_metadata = load_before_package(args.before_package_dir)
    vllm, vllm_metadata = load_vllm_experts(args.vllm_dir)
    torch.manual_seed(args.seed)
    e, h, i, r = args.experts, args.hidden, args.intermediate, args.top_k
    w1 = (torch.randn((e, 2*i, h), device="cuda", dtype=torch.bfloat16) / h**0.5).contiguous()
    w2 = (torch.randn((e, h, i), device="cuda", dtype=torch.bfloat16) / i**0.5).contiguous()
    q1, s1 = prepare_weight(w1, torch)
    q2, s2 = prepare_weight(w2, torch)
    cosmos_module, cosmos_metadata = load_cosmos_experts(args.cosmos_dir, e, h, i, r, "cuda", torch.bfloat16, w1, w2)
    cosmos = torch.compile(make_cosmos_forward(cosmos_module), fullgraph=True, dynamic=True)
    helpers = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "bench_moe_ffn.py", "bench_moe_cosmos.py", "bench_moe_vllm.py",
        "cosmos_moe_baseline.py", "vllm_moe_baseline.py", "vllm_moe_native.py")),
        ROOT / "tools/build_cutlass.py", ROOT / "tools/build_cutlass_moe.py"]
    state = gpu_state()
    package = Path(grain_gemm.__file__).resolve().parent
    report = dict(
        complete=False, timestamp_utc=datetime.now(timezone.utc).isoformat(), expected_tokens=args.tokens,
        seed=args.seed, dtype="bfloat16", group_size=256, activation="swiglu",
        experts=e, hidden=h, intermediate=i, top_k=r, methods=list(METHODS),
        gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
        software=dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda, python=platform.python_version()),
        rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup, warmup_ms=args.warmup_ms,
        timing="Full-call CUDA events including host dispatch gaps; rotating four-method order; median of per-round medians; no CUDA Graph capture",
        scope="Pre-routed full expert FFN including counts/alignment, gate/up, activation, down and combine; INT8 includes activation quantization. Excludes router/top-k, weight quantization/layout preparation, compilation, correctness, warmup and cooling. No serving-engine overhead.",
        throughput="6*T*top_k*H*I / (latency_seconds*1e12), valid-route dense operations; INT8 effective TOPS and BF16 effective TFLOPS. Padding and non-GEMM operations excluded.",
        score_policy="Normalize FP32 once, round to BF16; Cosmos receives BF16 and both INT8/vLLM receive FP32 with identical numeric values",
        numerical_difference="Both INT8 paths retain G256 integer-dot references and post-down FP32 combine; vLLM applies score to the FC2 accumulator before BF16 cast and sums in FP32; Cosmos compiled applies score before down with fused pointwise operations and BF16 scatter_add",
        correctness=dict(independent_cpu_tokens=args.reference_tokens, int8_rtol=0.015, int8_atol=0.025,
                         bf16_rtol=0.025, bf16_atol=0.008, before_after_full_output_check=True),
        cosmos_compilation=dict(fullgraph=True, dynamic=True, backend="inductor", mode="default"),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        before=before_metadata, after={**hash_package(package), "checkout": source_metadata(package)},
        vllm=vllm_metadata, omni=omni_metadata(args.omni_dir), cosmos=cosmos_metadata,
        benchmark_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in helpers},
        weights=dict(w1_shape=list(w1.shape), w1_stride=list(w1.stride()),
                     w2_shape=list(w2.shape), w2_stride=list(w2.stride()),
                     int8_w1_stride=list(q1.stride()), int8_w2_stride=list(q2.stride()),
                     scale_w1_shape=list(s1.shape), scale_w2_shape=list(s2.shape)),
        frequency_note=("GPU frequency reduced to 1.2 GHz; actual supported locked setting is 1197 MHz."
                        if state.get("min_freq_hz") == state.get("max_freq_hz") == 1197000000
                        else "Clocks unchanged by benchmark; see recorded snapshots."),
        memory_clock_policy="EMC policy unchanged; instantaneous current/min/max snapshots are recorded where exposed. EMC is not locked by this benchmark.",
        gpu_state_before=state, cases=[],
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
            config = vllm.config_for(tokens, w1, w2, r)
            functions = dict(
                int8_before=lambda: before_package.moe_ffn(x, q1, q2, scores, ids, scale_gate_up=s1, scale_down=s2),
                int8_auto=lambda: grain_gemm.moe_ffn(x, q1, q2, scores, ids, scale_gate_up=s1, scale_down=s2),
                vllm_bf16=lambda: vllm(x, w1, w2, scores, ids),
                cosmos_bf16_compiled=lambda: cosmos(x, ids, scores_bf16),
            )
            try:
                outputs, first_call = {}, {}
                for name, function in functions.items():
                    started = time.perf_counter()
                    outputs[name] = function()
                    torch.cuda.synchronize()
                    first_call[name] = time.perf_counter() - started
                    if not torch.isfinite(outputs[name]).all().item():
                        raise AssertionError(f"Nonfinite output from {name}")
                count = min(tokens, args.reference_tokens)
                checks = dict(independent_cpu_tokens=count, independent_cpu_routes=count*r, all_outputs_finite=True)
                expected = cpu_reference(x[:count], ids[:count], scores[:count], q1, q2, s1, s2, torch)
                for name in ("int8_before", "int8_auto"):
                    torch.testing.assert_close(outputs[name][:count].cpu(), expected, rtol=0.015, atol=0.025)
                    checks[name + "_vs_int32_reference"] = difference(outputs[name][:count].cpu(), expected, torch)
                expected = vllm_cpu_reference(x[:count], ids[:count], scores[:count], w1, w2, torch)
                torch.testing.assert_close(outputs["vllm_bf16"][:count].cpu(), expected, rtol=0.025, atol=0.008)
                checks["vllm_vs_cpu"] = difference(outputs["vllm_bf16"][:count].cpu(), expected, torch)
                expected = bf16_cpu_reference(x[:count], ids[:count], scores[:count], w1, w2, torch, fused_activation=True)
                torch.testing.assert_close(outputs["cosmos_bf16_compiled"][:count].cpu(), expected, rtol=0.025, atol=0.008)
                checks["cosmos_compiled_vs_cpu"] = difference(outputs["cosmos_bf16_compiled"][:count].cpu(), expected, torch)
                torch.testing.assert_close(outputs["int8_auto"], outputs["int8_before"], rtol=0.015, atol=0.025)
                dispatch = dict(
                    int8_before=record_dispatch(before_package, functions["int8_before"]),
                    int8_auto=record_dispatch(grain_gemm, functions["int8_auto"]),
                )
                for left, right in (("int8_auto", "int8_before"), ("int8_auto", "vllm_bf16"),
                                    ("int8_auto", "cosmos_bf16_compiled"), ("vllm_bf16", "cosmos_bf16_compiled")):
                    checks[left + "_vs_" + right] = difference(outputs[left], outputs[right], torch)
                torch.cuda.synchronize()
                cooled = wait_for_cooling(args)
                warmups = {name: warmup(function, args, torch) for name, function in functions.items()}
                medians, all_samples, orders, states = {name: [] for name in METHODS}, [], [], []
                for round_index in range(args.rounds):
                    names = list(METHODS)
                    offset = round_index % len(names)
                    order = names[offset:] + names[:offset]
                    orders.append(order)
                    current = {}
                    for name in order:
                        samples = []
                        for _ in range(args.repetitions):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            functions[name]()
                            end.record()
                            end.synchronize()
                            samples.append(start.elapsed_time(end))
                        current[name] = samples
                        medians[name].append(statistics.median(samples))
                    all_samples.append(current)
                    state = gpu_state()
                    states.append(state)
                    if state.get("sw_thermal_slowdown") == "Active" or state.get("temperature_c", 0) >= args.max_temperature:
                        raise RuntimeError(f"Thermally limited case T={tokens}: {state}")
                results = {name: summarize(values, 6*tokens*r*h*i, name.startswith("int8")) for name, values in medians.items()}
                latency = {name: result["median_ms"] for name, result in results.items()}
                after = gpu_state()
                verify_provenance(report, helpers)
                case = dict(tokens=tokens, experts=e, hidden=h, intermediate=i, top_k=r, vllm_config=config,
                            int8_dispatch=dispatch, provenance_unchanged=True,
                            route_counts=torch.bincount(ids.flatten(), minlength=e).cpu().tolist(), checks=checks,
                            first_call_seconds=first_call, warmup=warmups, results=results,
                            sample_values_ms=all_samples, measurement_orders=orders, round_gpu_states=states,
                            gpu_state_before=cooled, gpu_state_after=after,
                            ratios=dict(before_over_auto=latency["int8_before"]/latency["int8_auto"],
                                        vllm_bf16_over_auto=latency["vllm_bf16"]/latency["int8_auto"],
                                        cosmos_bf16_over_auto=latency["cosmos_bf16_compiled"]/latency["int8_auto"],
                                        cosmos_over_vllm=latency["cosmos_bf16_compiled"]/latency["vllm_bf16"]))
                report["cases"].append(case)
                report["complete"] = len(report["cases"]) == len(args.tokens)
                report["gpu_state_after"] = after
                save(report, args.output)
                print(f"T={tokens}: " + "; ".join(f"{name} {latency[name]:.3f} ms" for name in METHODS)
                      + f"; before/auto {case['ratios']['before_over_auto']:.2f}x", flush=True)
            except Exception as error:
                report["failure"] = dict(tokens=tokens, error=f"{type(error).__name__}: {error}")
                save(report, args.output)
                raise


if __name__ == "__main__":
    main()
