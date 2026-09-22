"""Benchmark prequantized GEMM; input preparation and quantization are not timed."""

import argparse
import json
import platform
from pathlib import Path


def check_sample(a, b, scale_a, scale_b, result, group_size):
    """Check up to 32 evenly spaced rows/columns using CPU INT32 dot products."""
    import torch

    rows = torch.linspace(
        0, a.shape[0] - 1, min(32, a.shape[0]), device=a.device
    ).long()
    columns = torch.linspace(
        0, b.shape[1] - 1, min(32, b.shape[1]), device=b.device
    ).long()
    a_sample = a.index_select(0, rows).cpu().to(torch.int32)
    b_sample = b.index_select(1, columns).cpu().to(torch.int32)
    sa = scale_a.index_select(0, rows).cpu()
    sb = scale_b.index_select(1, columns).cpu()
    expected = torch.zeros((len(rows), len(columns)), dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], group_size)):
        partial = (
            a_sample[:, start : start + group_size]
            @ b_sample[start : start + group_size]
        )
        expected += partial.float() * (sa[:, group, None] * sb[group, None, :])
    actual = result.index_select(0, rows).index_select(1, columns).cpu()
    expected = expected.to(actual.dtype)
    tolerances = {
        torch.float32: {"rtol": 1e-4, "atol": 1e-3},
        torch.float16: {"rtol": 1e-3, "atol": 1e-3},
        torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
    }
    torch.testing.assert_close(actual, expected, **tolerances[actual.dtype])
    if not torch.isfinite(result).all().item():
        raise AssertionError("The output contains non-finite values")
    return {
        "checked_rows": len(rows),
        "checked_columns": len(columns),
        "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-size", type=int, nargs="+", choices=[32, 64, 128, 256], default=[256])
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="output dtype")
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--rep-ms", type=int, default=100, help="target duration of each captured timing batch")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="also write the complete result as JSON")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.rep_ms) <= 0:
        parser.error("M, N, K, and rep-ms must be positive")

    import torch
    import triton
    from triton.testing import do_bench_cudagraph

    from grain_gemm.baselines import sglang_int8_gemm

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    a = torch.randint(-128, 128, (args.m, args.k), dtype=torch.int8, device="cuda")
    # A conventional [N, K] weight tensor is exposed as logical B[K, N].
    b = torch.randint(-128, 128, (args.n, args.k), dtype=torch.int8, device="cuda").t()
    config = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
    }
    report = {
        "baseline": "sglang_int8_gemm",
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
        },
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "layout": {"a": "row-major", "b": "column-major", "scales": "row-major"},
        "output_dtype": args.dtype,
        "seed": args.seed,
        "launch_config": config,
        "timing": {
            "method": "triton.testing.do_bench_cudagraph",
            "statistic": "median",
            "rep_ms": args.rep_ms,
            "scope": "prequantized GEMM only; compilation, warmup, input preparation, and quantization excluded",
            "cache_policy": "repeated inputs; no cache flush between launches",
        },
        "results": [],
    }
    for group_size in args.group_size:
        groups = (args.k + group_size - 1) // group_size
        scale_a = 0.001 + torch.rand((args.m, groups), device="cuda") * 0.01
        scale_b = 0.001 + torch.rand((groups, args.n), device="cuda") * 0.01

        def run():
            return sglang_int8_gemm(
                a, b, scale_a, scale_b,
                group_size=group_size, output_dtype=dtype, **config,
            )

        # Compilation and the independent correctness check precede all timing.
        result = run()
        correctness = check_sample(a, b, scale_a, scale_b, result, group_size)
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        milliseconds = do_bench_cudagraph(run, rep=args.rep_ms, return_mode="median")
        report["results"].append({
            "group_size": group_size,
            "block_k": group_size,
            "latency_ms": milliseconds,
            "effective_int8_tops": 2 * args.m * args.n * args.k / (milliseconds * 1e9),
            "correctness": correctness,
        })

    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
