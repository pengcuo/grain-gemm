"""Compare GrainGEMM, the SGLang INT8 baseline, and torch.mm BF16 per square size.

Inputs are quantized before timing. Every reported latency includes the complete
GEMM, including group scaling and BF16 output, but excludes input preparation,
compilation, autotuning, warmup, and host dispatch. JSON and CSV are updated after
each size so an interrupted sweep retains its completed measurements.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess


IMPLEMENTATIONS = ("grain_int8_gemm", "sglang_int8_gemm", "torch_mm_bf16")
SGLANG_CONFIG = {"block_m": 32, "block_n": 64, "num_warps": 4, "num_stages": 2}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def quantize_rows(x, group_size):
    """Use the same symmetric, per-row K-group quantizer as compare_bf16.py."""
    import torch

    blocks = x.float().reshape(x.shape[0], -1, group_size)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-12) / 127
    quantized = (blocks / scales[..., None]).round().clamp(-127, 127)
    return quantized.to(torch.int8).reshape(x.shape), scales


def summarize(samples, operations, *, integer):
    median = statistics.median(samples)
    return {
        "round_medians_ms": samples,
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "effective_tops" if integer else "effective_tflops":
            operations / (median * 1e9),
    }


def export(report, output):
    """Replace each output atomically; never mark an unfinished sweep complete."""
    output.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_utc"] = utc_now()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)

    csv_path = output.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        fields = ["m", "n", "k", "group_size"]
        for name in IMPLEMENTATIONS:
            unit = "tflops" if name == "torch_mm_bf16" else "tops"
            fields.extend([
                f"{name}_median_us", f"{name}_min_us", f"{name}_max_us",
                f"{name}_effective_{unit}", f"{name}_round_medians_ms",
            ])
        fields.extend([
            "grain_speedup_over_bf16", "grain_speedup_over_sglang",
            "sglang_speedup_over_bf16", "grain_kernel_config",
        ])
        writer.writerow(fields)
        for case in report["reports"]:
            row = [
                case["shape"]["m"], case["shape"]["n"], case["shape"]["k"],
                case["group_size"],
            ]
            for name in IMPLEMENTATIONS:
                result = case["results"][name]
                key = "effective_tflops" if name == "torch_mm_bf16" else "effective_tops"
                row.extend([
                    result["median_ms"] * 1000, result["min_ms"] * 1000,
                    result["max_ms"] * 1000, result[key],
                    json.dumps(result["round_medians_ms"]),
                ])
            row.extend([
                case["grain_speedup_over_bf16"], case["grain_speedup_over_sglang"],
                case["sglang_speedup_over_bf16"],
                json.dumps(case["grain_kernel_config"], sort_keys=True),
            ])
            writer.writerow(row)
    temporary_csv.replace(csv_path)


def source_metadata(grain_gemm):
    script = Path(__file__).resolve()
    repository = script.parent.parent

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *arguments],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    package = Path(grain_gemm.__file__).resolve().parent
    sources = {
        f"grain_gemm/{path.relative_to(package).as_posix()}":
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".cu", ".json")
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "comparison_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "check_sample_script_sha256": hashlib.sha256(
            script.with_name("bench_sglang.py").read_bytes()
        ).hexdigest(),
        "package_source_sha256": sources,
    }


def run_case(size, args, torch, grain_gemm):
    from triton.testing import do_bench_cudagraph

    from bench_sglang import check_sample
    from grain_gemm.baselines import sglang_int8_gemm

    # Reset for every size: a case's inputs do not depend on the sweep order.
    torch.manual_seed(args.seed)
    a_bf16 = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    weight_bf16 = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    b_bf16 = weight_bf16.T
    a_int8, scale_a = quantize_rows(a_bf16, args.group_size)
    weight_int8, weight_scales = quantize_rows(weight_bf16, args.group_size)
    b_int8 = weight_int8.T
    scale_b = weight_scales.T.contiguous()

    def run_grain():
        return grain_gemm.int8_gemm(
            a_int8, b_int8, scale_a, scale_b,
            group_size=args.group_size, output_dtype=torch.bfloat16,
        )

    def run_sglang():
        return sglang_int8_gemm(
            a_int8, b_int8, scale_a, scale_b,
            group_size=args.group_size, output_dtype=torch.bfloat16,
            **SGLANG_CONFIG,
        )

    def run_bf16():
        return torch.mm(a_bf16, b_bf16)

    functions = dict(zip(IMPLEMENTATIONS, (run_grain, run_sglang, run_bf16)))
    # Compile/select kernels and check each INT8 implementation independently,
    # before entering any timed graph. Agreement with the baseline alone is
    # insufficient: check_sample computes exact CPU INT32 sums per K-group.
    bf16_result = run_bf16()
    if not torch.isfinite(bf16_result).all().item():
        raise AssertionError("torch.mm returned non-finite output")
    reference_rms_squared = bf16_result.float().square().mean()
    correctness = {}
    quality = {}
    for name in IMPLEMENTATIONS[:2]:
        result = functions[name]()
        if result.dtype != torch.bfloat16 or result.shape != (size, size):
            raise AssertionError(f"{name} must return a [{size}, {size}] BF16 tensor")
        correctness[name] = check_sample(
            a_int8, b_int8, scale_a, scale_b, result, args.group_size
        )
        difference = result.float() - bf16_result.float()
        quality[name] = {
            "relative_rms_error": (
                difference.square().mean() / reference_rms_squared
            ).sqrt().item(),
            "max_abs_error": difference.abs().max().item(),
        }
        del result, difference
    del bf16_result, reference_rms_squared

    for _ in range(args.warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()

    config_function = getattr(grain_gemm, "get_kernel_config", None)
    grain_config = (
        config_function(a_int8, b_int8, group_size=args.group_size,
                        output_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b)
        if callable(config_function) else None
    )
    measurements = {name: [] for name in IMPLEMENTATIONS}
    orders = []
    for index in range(args.rounds):
        offset = index % len(IMPLEMENTATIONS)
        order = list(IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset])
        orders.append(order)
        for name in order:
            ms = do_bench_cudagraph(
                functions[name], rep=args.rep_ms, return_mode="median"
            )
            measurements[name].append(ms)
        print(f"  Finished timing round {index + 1}/{args.rounds}", flush=True)

    summaries = {
        name: summarize(samples, 2 * size**3, integer=name != "torch_mm_bf16")
        for name, samples in measurements.items()
    }
    grain_ms = summaries["grain_int8_gemm"]["median_ms"]
    sglang_ms = summaries["sglang_int8_gemm"]["median_ms"]
    bf16_ms = summaries["torch_mm_bf16"]["median_ms"]
    return {
        "timestamp_utc": utc_now(),
        "shape": {"m": size, "n": size, "k": size},
        "group_size": args.group_size,
        "seed": args.seed,
        "output_dtype": "bfloat16 for all implementations",
        "layout": {"a": "row-major", "b": "column-major", "scales": "row-major"},
        "input_strides": {
            "a": list(a_int8.stride()), "b": list(b_int8.stride()),
            "scale_a": list(scale_a.stride()), "scale_b": list(scale_b.stride()),
        },
        "grain_kernel_config": grain_config,
        "sglang_config": SGLANG_CONFIG,
        "sglang_config_autotuned": False,
        "measurement_order": orders,
        "sampled_correctness": correctness,
        "quantization_difference": {
            "reference": "torch.mm on the original BF16 inputs",
            "note": "Synthetic-input quantization difference, not model accuracy or kernel correctness.",
            "results": quality,
        },
        "results": summaries,
        "grain_speedup_over_bf16": bf16_ms / grain_ms,
        "grain_speedup_over_sglang": sglang_ms / grain_ms,
        "sglang_speedup_over_bf16": bf16_ms / sglang_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(range(1024, 16385, 1024)),
                        help="independent square cases, each with M=N=K=SIZE")
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128, 256], default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100,
                        help="target duration of each captured timing batch, not the entire round")
    parser.add_argument("--warmup", type=int, default=50, help="untimed warmup calls per implementation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True,
                        help="aggregate .json file; also writes a matching .csv")
    args = parser.parse_args()
    if min(*args.sizes, args.rounds, args.rep_ms, args.warmup) <= 0:
        parser.error("sizes, rounds, rep-ms, and warmup must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must not contain duplicates")
    if any(size % args.group_size for size in args.sizes):
        parser.error("this comparison's quantizer requires every size divisible by G")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix; a matching .csv is also written")

    import torch
    import triton
    import grain_gemm

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    if not callable(getattr(grain_gemm, "int8_gemm", None)):
        parser.error("this GrainGEMM installation does not export int8_gemm")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    report = {
        "benchmark": "GrainGEMM and SGLang INT8 versus PyTorch BF16 square GEMM sweep",
        "started_at_utc": utc_now(),
        "expected_sizes": args.sizes,
        "group_size": args.group_size,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_properties": {
            "cuda_device_index": device,
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "driver": driver, "grain_gemm": getattr(grain_gemm, "__version__", None),
        },
        "source": source_metadata(grain_gemm),
        "torch_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "timing": {
            "method": "triton.testing.do_bench_cudagraph",
            "statistic": "median of per-round medians, separately for each size",
            "rounds": args.rounds, "rep_ms": args.rep_ms,
            "warmup_calls_per_implementation": args.warmup,
            "measurement_order": "rotates implementation order each round; exact orders saved per case",
            "scope": "GPU GEMM execution including group scaling, accumulation, and BF16 output",
            "includes_input_quantization": False,
            "includes_compilation_or_autotuning": False,
            "includes_warmup": False,
            "includes_host_dispatch": False,
            "cache_policy": "repeated inputs; no cache flush",
            "rep_ms_note": "Target duration of each captured batch; graph replay count follows installed Triton.",
        },
        "quantization": {
            "input_dtype": "bfloat16",
            "quantized_dtype": "int8",
            "scale_dtype": "float32",
            "method": "per row/column and K-group: max(abs(x)).clamp_min(1e-12)/127; round and clamp to [-127,127]",
            "performed_before_timing": True,
        },
        "complete": False,
        "reports": [],
    }
    export(report, args.output)
    for index, size in enumerate(args.sizes, 1):
        print(f"[{index}/{len(args.sizes)}] M=N=K={size}, G={args.group_size}", flush=True)
        try:
            case = run_case(size, args, torch, grain_gemm)
        except Exception as error:
            report["failure"] = {"size": size, "type": type(error).__name__, "message": str(error)}
            export(report, args.output)
            raise
        report["reports"].append(case)
        export(report, args.output)
        grain = case["results"]["grain_int8_gemm"]
        sglang = case["results"]["sglang_int8_gemm"]
        bf16 = case["results"]["torch_mm_bf16"]
        print(
            f"  {size}: Grain {grain['median_ms'] * 1000:.2f} us ({grain['effective_tops']:.2f} TOPS)"
            f" | SGLang {sglang['median_ms'] * 1000:.2f} us ({sglang['effective_tops']:.2f} TOPS)"
            f" | BF16 {bf16['median_ms'] * 1000:.2f} us ({bf16['effective_tflops']:.2f} TFLOPS)"
            f" | Grain speedup: {case['grain_speedup_over_bf16']:.3f}x vs BF16,"
            f" {case['grain_speedup_over_sglang']:.3f}x vs SGLang",
            flush=True,
        )
    report["complete"] = True
    report["completed_at_utc"] = utc_now()
    export(report, args.output)
    print(f"Saved {len(report['reports'])} independent comparisons to {args.output} and {args.output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
