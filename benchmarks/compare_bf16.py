"""Compare prequantized SGLang INT8 GEMM with torch.mm on BF16 inputs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128, 256], default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.rounds, args.rep_ms) <= 0:
        parser.error("dimensions, rounds, and rep-ms must be positive")
    if args.k % args.group_size:
        parser.error("this comparison's input quantizer requires K divisible by G")

    import torch
    import triton
    from triton.testing import do_bench_cudagraph

    from bench_sglang import check_sample
    from grain_gemm.baselines import sglang_int8_gemm

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    torch.manual_seed(args.seed)

    def quantize_rows(x):
        blocks = x.float().reshape(x.shape[0], -1, args.group_size)
        scales = blocks.abs().amax(dim=-1).clamp_min(1e-12) / 127
        quantized = (blocks / scales[..., None]).round().clamp(-127, 127)
        return quantized.to(torch.int8).reshape(x.shape), scales

    # Both paths start with the same BF16 matrices. Quantization is not timed.
    a_bf16 = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    weight_bf16 = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
    b_bf16 = weight_bf16.T
    a_int8, scale_a = quantize_rows(a_bf16)
    weight_int8, weight_scales = quantize_rows(weight_bf16)
    b_int8 = weight_int8.T
    scale_b = weight_scales.T.contiguous()
    config = {"block_m": 32, "block_n": 64, "num_warps": 4, "num_stages": 2}

    def run_int8():
        return sglang_int8_gemm(
            a_int8, b_int8, scale_a, scale_b,
            group_size=args.group_size, output_dtype=torch.bfloat16, **config,
        )

    def run_bf16():
        return torch.mm(a_bf16, b_bf16)

    result_int8, result_bf16 = run_int8(), run_bf16()
    correctness = check_sample(
        a_int8, b_int8, scale_a, scale_b, result_int8, args.group_size
    )
    if not torch.isfinite(result_bf16).all().item():
        raise AssertionError("torch.mm returned non-finite output")
    difference = result_int8.float() - result_bf16.float()
    quality = {
        "reference": "torch.mm on the original BF16 inputs",
        "relative_rms_error": (
            difference.square().mean() / result_bf16.float().square().mean()
        ).sqrt().item(),
        "max_abs_error": difference.abs().max().item(),
        "note": "Synthetic-input quantization difference, not a model-accuracy test.",
    }
    del difference, result_int8, result_bf16
    for _ in range(50):
        run_int8()
        run_bf16()
    torch.cuda.synchronize()

    functions = {"sglang_int8_gemm": run_int8, "torch_mm_bf16": run_bf16}
    measurements = {name: [] for name in functions}
    orders = []
    for index in range(args.rounds):
        order = list(functions) if index % 2 == 0 else list(reversed(functions))
        orders.append(order)
        for name in order:
            ms = do_bench_cudagraph(
                functions[name], rep=args.rep_ms, return_mode="median"
            )
            measurements[name].append(ms)
        print(f"Finished timing round {index + 1}/{args.rounds}", flush=True)

    summaries = {}
    operations = 2 * args.m * args.n * args.k
    for name, samples in measurements.items():
        median = statistics.median(samples)
        summaries[name] = {
            "round_medians_ms": samples,
            "median_ms": median,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "effective_tops" if name == "sglang_int8_gemm" else "effective_tflops":
                operations / (median * 1e9),
        }

    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "driver": driver,
        },
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "group_size": args.group_size,
        "seed": args.seed,
        "output_dtype": "bfloat16 for both implementations",
        "layout": {"a": "row-major", "b": "column-major", "scales": "row-major"},
        "int8_config": config,
        "int8_config_autotuned": False,
        "torch_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "timing": {
            "method": "triton.testing.do_bench_cudagraph",
            "statistic": "median of per-round medians",
            "rounds": args.rounds, "rep_ms": args.rep_ms,
            "measurement_order": orders,
            "scope": "GPU GEMM execution only; excludes input quantization, compilation, warmup, and host dispatch overhead",
            "cache_policy": "repeated inputs; no cache flush",
        },
        "int8_sampled_correctness": correctness,
        "quantization_difference": quality,
        "results": summaries,
        "int8_speedup_over_bf16": (
            summaries["torch_mm_bf16"]["median_ms"]
            / summaries["sglang_int8_gemm"]["median_ms"]
        ),
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
