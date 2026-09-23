"""Offline GB10 G256 candidate search; no tuning is performed during inference.

Run before compare_backends.py, which measures the selected CuTe and Triton
configurations again against fresh BF16 results. This search does not establish
a global performance optimum. Native candidates require tools/build_cuda.py.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from _device import require_gb10
TRITON_CANDIDATES = [
    dict(backend="triton", block_m=64, block_n=64, num_warps=4, num_stages=2, swizzle=8),
    dict(backend="triton", block_m=64, block_n=128, num_warps=8, num_stages=3, swizzle=8),
    dict(backend="triton", block_m=64, block_n=128, num_warps=4, num_stages=3, swizzle=8),
    dict(backend="triton", block_m=128, block_n=64, num_warps=4, num_stages=3, swizzle=8),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(range(1024, 16385, 1024)))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-output", type=Path)
    args = parser.parse_args()
    if (min(*args.sizes, args.rounds, args.rep_ms) <= 0
            or any(n % 256 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes)):
        parser.error("sizes must be distinct positive multiples of 256; timing parameters must be positive")
    import torch
    require_gb10(torch)
    import triton
    from triton.testing import do_bench_cudagraph
    from grain_gemm.kernels.triton import launch
    from grain_gemm.kernels import cuda
    from bench_sglang import check_sample

    candidates = list(TRITON_CANDIDATES)
    if cuda.is_available():
        candidates += [dict(backend="cuda", config_id=i) for i in range(len(cuda.CONFIGS))]
    report = dict(
        complete=False, gpu=torch.cuda.get_device_name(), compute_capability=[12, 1],
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), expected_sizes=args.sizes,
        group_size=256, output_dtype="bfloat16", rounds=args.rounds, rep_ms=args.rep_ms,
        seed=2026, candidates=candidates, cases=[],
        scope="Prequantized GEMM; group scaling and output conversion included. Input preparation, compilation and warmup excluded.",
        timing="CUDA graphs, median of per-round medians, rotating candidate order, repeated inputs, no cache flush",
        native_configurations=cuda.CONFIGS,
    )
    package = Path(cuda.__file__).resolve().parent
    report["kernel_source_sha256"] = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (package / "triton.py", package / "cuda.py", package / "csrc" / "grain_cute.cu")
    }
    report["tuning_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["device_check_sha256"] = hashlib.sha256(Path(require_gb10.__code__.co_filename).read_bytes()).hexdigest()
    if cuda.is_available():
        report["native_build"] = json.loads((package / "_native" / "build.json").read_text())
    dispatch = dict(gpu="NVIDIA GB10", compute_capability=[12, 1], group_size=256,
                    output_dtype="bfloat16", square_configs={})
    for n in args.sizes:
        torch.manual_seed(2026)
        a = torch.randint(-128, 128, (n, n), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (n, n), device="cuda", dtype=torch.int8).T
        sa = torch.rand((n, n // 256), device="cuda") * .01
        sb = torch.rand((n // 256, n), device="cuda") * .01
        out = torch.empty((n, n), device="cuda", dtype=torch.bfloat16)
        funcs, checks = [], []
        for config in candidates:
            if config["backend"] == "cuda":
                def run(cfg=config):
                    cuda.launch(a, b, sa, sb, out, cfg["config_id"])
            else:
                def run(cfg=config):
                    launch(a, b, sa, sb, out, 256, cfg)
            run()
            checks.append(check_sample(a, b, sa, sb, out, 256))
            funcs.append(run)
        samples = [[] for _ in candidates]
        for round_index in range(args.rounds):
            for step in range(len(candidates)):
                index = (step + round_index) % len(candidates)
                samples[index].append(do_bench_cudagraph(funcs[index], rep=args.rep_ms, return_mode="median"))
        rows = [dict(config=config, round_medians_ms=timings,
                     median_ms=statistics.median(timings), correctness=check)
                for config, timings, check in zip(candidates, samples, checks)]
        best = min(rows, key=lambda row: row["median_ms"])
        # Keep the best Triton candidate for installations without a native build.
        best_triton = min((row for row in rows if row["config"]["backend"] == "triton"),
                          key=lambda row: row["median_ms"])
        report["cases"].append(dict(size=n, candidates=rows, selected=best["config"]))
        dispatch["square_configs"][str(n)] = dict(selected=best["config"], triton=best_triton["config"])
        native_rows = [row for row in rows if row["config"]["backend"] == "cuda"]
        if native_rows:
            best_cuda = min(native_rows, key=lambda row: row["median_ms"])
            dispatch["square_configs"][str(n)]["cuda"] = best_cuda["config"]
        print(f"{n}: {best['median_ms'] * 1000:.2f} us {best['config']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del a, b, sa, sb, out, funcs
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.dispatch_output:
        dispatch["tuning_report_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        # This search measures square shapes only; retain independently measured
        # rectangular entries when updating the same device/format table.
        if args.dispatch_output.exists():
            previous = json.loads(args.dispatch_output.read_text())
            if all(previous.get(key) == dispatch[key] for key in (
                "gpu", "compute_capability", "group_size", "output_dtype"
            )):
                if "rectangular_configs" in previous:
                    dispatch["rectangular_configs"] = previous["rectangular_configs"]
                # The separate collective benchmark measures CUTLASS entries.
                for size, entry in dispatch["square_configs"].items():
                    old_entry = previous.get("square_configs", {}).get(size, {})
                    for key in ("cutlass", "cutlass_measurement_sha256"):
                        if key in old_entry:
                            entry[key] = old_entry[key]
        args.dispatch_output.parent.mkdir(parents=True, exist_ok=True)
        args.dispatch_output.write_text(json.dumps(dispatch, indent=2) + "\n")


if __name__ == "__main__":
    main()
