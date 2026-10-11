"""Compare Thor CuTe and Triton INT8 against BF16 using the GB10 protocol.

Defaults match compare_backends.py: M=N=K=1024..16384 in steps of 1024,
G256, BF16 output, seed 2026, five rounds, 100 ms captured batches and
50 warmup calls per implementation. Quantization is outside the GPU timer.
"""

import argparse
import hashlib
from pathlib import Path
import platform
import re
import subprocess

import compare_backends as shared


DEFAULT_SIZES = list(range(1024, 16385, 1024))


def require_thor(torch, device=None):
    """Check the device before loading a native library or measuring."""
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires NVIDIA Thor SM110 and CUDA-enabled PyTorch")
    name = torch.cuda.get_device_name(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (11, 0) or re.search(r"\b(?:Thor|T5000)\b", name, re.IGNORECASE) is None:
        raise RuntimeError(
            f"This comparison requires NVIDIA Thor (SM110 / compute capability 11.0); "
            f"detected {name} with capability {capability}"
        )
    return {"name": name, "compute_capability": list(capability)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="independent square cases, each with M=N=K=SIZE")
    parser.add_argument("--group-size", type=int, choices=[256], default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100,
                        help="target duration of each captured timing batch, not the entire round")
    parser.add_argument("--warmup", type=int, default=50,
                        help="untimed warmup calls per implementation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True,
                        help="aggregate .json file; also writes a matching .csv")
    args = parser.parse_args(argv)
    if min(*args.sizes, args.rounds, args.rep_ms, args.warmup) <= 0:
        parser.error("sizes, rounds, rep-ms, and warmup must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must not contain duplicates")
    if any(size % args.group_size for size in args.sizes):
        parser.error("native G256 comparison requires every size divisible by 256")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix; a matching .csv is also written")
    return args


def source_metadata(grain_gemm):
    """Retain the shared runner's provenance and include Thor pipeline headers."""
    metadata = shared.source_metadata(grain_gemm)
    metadata["shared_comparison_script_sha256"] = metadata["comparison_script_sha256"]
    metadata["comparison_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # The Thor device guard lives in this entry point, rather than _device.py.
    metadata["device_check_sha256"] = metadata["comparison_script_sha256"]
    package = Path(grain_gemm.__file__).resolve().parent
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".cuh", ".h", ".hpp"):
            key = f"grain_gemm/{path.relative_to(package).as_posix()}"
            metadata["package_source_sha256"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata


def gpu_state():
    """Record the configured GPC clock and the driver's thermal status."""
    directory = Path("/sys/class/devfreq/gpu-gpc-0")
    clocks = {}
    for name in ("governor", "min_freq", "max_freq", "cur_freq"):
        path = directory / name
        if path.is_file():
            value = path.read_text().strip()
            clocks[name] = value if name == "governor" else int(value)
    values = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=temperature.gpu,clocks.current.graphics,clocks_event_reasons.sw_thermal_slowdown",
         "--format=csv,noheader,nounits"],
        text=True, stderr=subprocess.DEVNULL,
    ).strip().split(",")
    return {
        "timestamp_utc": shared.utc_now(),
        "gpc_devfreq_hz": clocks,
        "temperature_c": float(values[0].strip()),
        "graphics_clock_mhz": int(values[1].strip()),
        "thermal_slowdown": values[2].strip(),
    }


def main(argv=None):
    args = parse_args(argv)
    import torch
    require_thor(torch)
    import triton
    import grain_gemm
    from grain_gemm.kernels import cuda

    if not cuda.is_available():
        raise RuntimeError(
            "Build the Thor CuTe library first: python tools/build_cuda.py "
            "--cutlass-dir /path/to/cutlass --arch sm_110"
        )
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
        "benchmark": "Thor forced GrainGEMM CUDA/CuTe and Triton INT8 versus PyTorch BF16 square GEMM sweep",
        "forced_backends": shared.INT8_BACKENDS,
        "protocol_reference": "benchmarks/compare_backends.py (GB10)",
        "configuration_policy": "Installed Thor dispatch table; compatible defaults for unmeasured sizes. No offline tuning in this comparison.",
        "started_at_utc": shared.utc_now(),
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
        "gpu_state_before": gpu_state(),
        "torch_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "timing": {
            "method": "triton.testing.do_bench_cudagraph",
            "statistic": "median of per-round medians, separately for each size",
            "rounds": args.rounds, "rep_ms": args.rep_ms,
            "warmup_calls_per_implementation": args.warmup,
            "measurement_order": "rotates implementation order each round; exact orders saved per case",
            "scope": "GPU GEMM execution including group scaling, accumulation, and BF16 output",
            "correctness_checks_performed_before_timing": True,
            "includes_input_quantization": False,
            "includes_compilation_or_autotuning": False,
            "includes_warmup": False,
            "includes_host_dispatch": False,
            "cache_policy": "repeated inputs; no cache flush",
            "rep_ms_note": "Target duration of each captured batch; graph replay count follows installed Triton.",
        },
        "quantization": {
            "input_dtype": "bfloat16", "quantized_dtype": "int8", "scale_dtype": "float32",
            "method": "per row/column and K-group: max(abs(x)).clamp_min(1e-12)/127; round and clamp to [-127,127]",
            "performed_before_timing": True,
        },
        "complete": False,
        "reports": [],
    }
    shared.export(report, args.output)
    for index, size in enumerate(args.sizes, 1):
        print(f"[{index}/{len(args.sizes)}] Thor M=N=K={size}, G={args.group_size}", flush=True)
        try:
            before = gpu_state()
            # Share the actual measurement and checks, not just their defaults.
            case = shared.run_case(size, args, torch, grain_gemm)
            case["gpu_state_before"] = before
            case["gpu_state_after"] = gpu_state()
            if case["gpu_state_after"]["thermal_slowdown"] == "Active":
                raise RuntimeError("Thermal slowdown occurred; cool the GPU before repeating this comparison")
        except Exception as error:
            report["failure"] = {"size": size, "type": type(error).__name__, "message": str(error)}
            shared.export(report, args.output)
            raise
        report["reports"].append(case)
        shared.export(report, args.output)
        cute = case["results"]["cute_int8_gemm"]
        triton_result = case["results"]["triton_int8_gemm"]
        bf16 = case["results"]["torch_mm_bf16"]
        print(
            f"  {size}: CuTe {cute['median_ms'] * 1000:.2f} us ({cute['effective_tops']:.2f} TOPS)"
            f" | Triton {triton_result['median_ms'] * 1000:.2f} us ({triton_result['effective_tops']:.2f} TOPS)"
            f" | BF16 {bf16['median_ms'] * 1000:.2f} us ({bf16['effective_tflops']:.2f} TFLOPS)"
            f" | CuTe speedup: {case['cute_speedup_over_bf16']:.3f}x vs BF16,"
            f" {case['cute_speedup_over_triton']:.3f}x vs Triton",
            flush=True,
        )
    report["complete"] = True
    report["gpu_state_after"] = gpu_state()
    report["completed_at_utc"] = shared.utc_now()
    shared.export(report, args.output)
    print(f"Saved {len(report['reports'])} comparisons to {args.output} and {args.output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
