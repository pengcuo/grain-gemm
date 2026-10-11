"""Tune and independently compare SM110 G256 kernels on NVIDIA Thor.

Example: PYTHONPATH=src python benchmarks/bench_thor.py --tune
Then: PYTHONPATH=src python benchmarks/bench_thor.py
Timing includes group scaling and BF16 conversion, excludes input quantization.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = [
    # 16384 cubed is omitted after instability on this Thor device.
    (n, n, n) for n in (1024, 2048, 3072, 4096, 6144, 8192, 12288)
] + [
    (256, 1024, 1536),
    (256, 1536, 1024),
    (256, 2048, 1536),
    (1024, 1024, 1536),
    (1024, 1536, 1024),
    (1024, 4096, 1536),
    (2048, 2048, 1536),
    (2048, 151936, 1536),
    (512, 3072, 1536),
    (4096, 3072, 1536),
    (1536, 1024, 4096),
]
TRITON_CONFIGS = [
    dict(backend="triton", block_m=m, block_n=n, num_warps=w, num_stages=s, swizzle=8)
    for m, n, w, s in (
        (32, 64, 4, 2),
        (64, 64, 4, 2),
        (64, 128, 8, 3),
        (64, 128, 4, 3),
        (128, 64, 4, 3),
        (64, 128, 4, 2),
        (128, 128, 8, 3),
        (128, 128, 4, 3),
    )
]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--reuse-fallbacks",
        action="store_true",
        help="Remeasure prior CUDA/Triton winners alongside new SM110 CuTe and CUTLASS candidates",
    )
    parser.add_argument(
        "--shapes", nargs="+", help="M x N x K tuples, e.g. 256x1024x1536"
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--rep-ms", type=int)
    parser.add_argument("--warmup-ms", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dispatch-output",
        type=Path,
        default=ROOT / "src/grain_gemm/kernels/configs/thor_sm110_g256.json",
    )
    args = parser.parse_args()
    if args.reuse_fallbacks and not args.tune:
        parser.error("--reuse-fallbacks requires --tune")
    fallback_table = None
    if args.reuse_fallbacks:
        fallback_table = json.loads(args.dispatch_output.read_text())
        if (
            fallback_table.get("compute_capability") != [11, 0]
            or fallback_table.get("group_size") != 256
        ):
            parser.error("Fallback table must contain Thor SM110 G256 measurements")
    shapes = (
        [tuple(map(int, s.split("x"))) for s in args.shapes]
        if args.shapes
        else DEFAULT_SHAPES
    )
    if any(
        len(s) != 3 or min(s) <= 0 or s[0] % 64 or s[1] % 64 or s[2] % 256
        for s in shapes
    ):
        parser.error(
            "M/N must be positive multiples of 64; K must be a positive multiple of 256"
        )
    args.rounds = args.rounds if args.rounds is not None else (3 if args.tune else 5)
    args.rep_ms = args.rep_ms if args.rep_ms is not None else (20 if args.tune else 100)
    if args.rounds <= 0 or args.rep_ms <= 0 or args.warmup_ms < 0:
        parser.error(
            "rounds and rep-ms must be positive; warmup-ms must be nonnegative"
        )
    output = args.output or ROOT / "benchmarks/results/thor" / (
        "tuning.json" if args.tune else "comparison.json"
    )
    import torch, triton
    from triton.testing import do_bench_cudagraph
    from grain_gemm import int8_gemm, get_kernel_config
    from grain_gemm.kernels import cuda, cutlass
    from grain_gemm.kernels.triton import launch
    from bench_sglang import check_sample
    from compare_backends import quantize_rows, check_graph_replay

    torch.set_num_threads(4)
    if (
        torch.cuda.get_device_capability() != (11, 0)
        or "Thor" not in torch.cuda.get_device_name()
    ):
        parser.error("This benchmark and dispatch table require NVIDIA Thor SM110")
    if not cuda.is_available():
        parser.error(
            "Build the SM110 CUDA backend first with tools/build_cuda.py --arch sm_110"
        )
    package = ROOT / "src/grain_gemm"
    report = dict(
        complete=False,
        mode="tuning" if args.tune else "comparison",
        gpu=torch.cuda.get_device_name(),
        compute_capability=[11, 0],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        expected_shapes=shapes,
        software=dict(
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
        ),
        seed=2026,
        rounds=args.rounds,
        rep_ms=args.rep_ms,
        warmup_ms=args.warmup_ms,
        reuse_fallbacks=args.reuse_fallbacks,
        fallback_table_sha256=(
            hashlib.sha256(args.dispatch_output.read_bytes()).hexdigest()
            if args.reuse_fallbacks
            else None
        ),
        timing="CUDA graphs; rotating order; median of round medians; repeated inputs; no cache flush",
        scope="Prequantized G256 GEMM including scaling and BF16 conversion; excludes quantization, compilation, warmup and host dispatch",
        scale_layout=dict(a="FP32 contiguous [M,K/256]", b="FP32 contiguous [K/256,N]"),
        source_hashes={
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob("*")
            if p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".json") and p.is_file()
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_library_sha256=hashlib.sha256(cuda._LIBRARY.read_bytes()).hexdigest(),
        native_build=json.loads(cuda._LIBRARY.with_name("build.json").read_text()),
        cutlass_library_sha256=(
            hashlib.sha256(cutlass._LIBRARY.read_bytes()).hexdigest()
            if cutlass.is_available()
            else None
        ),
        cutlass_build=(
            json.loads(cutlass._LIBRARY.with_name("cutlass_build.json").read_text())
            if cutlass.is_available()
            else None
        ),
        gpu_properties=dict(
            multiprocessor_count=torch.cuda.get_device_properties(
                0
            ).multi_processor_count,
            shared_memory_per_block_optin=torch.cuda.get_device_properties(
                0
            ).shared_memory_per_block_optin,
        ),
        cases=[],
    )
    dispatch = dict(
        gpu=torch.cuda.get_device_name(),
        compute_capability=[11, 0],
        group_size=256,
        output_dtype="bfloat16",
        square_configs={},
        rectangular_configs={},
    )
    for m, n, k in shapes:
        torch.manual_seed(2026)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        a, sa = quantize_rows(x, 256)
        weights, scales = quantize_rows(w, 256)
        b = weights.T
        sb = scales.T.contiguous()
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        configs = []
        functions = []
        checks = []
        if args.tune:
            previous_entry = None
            if fallback_table:
                previous_entry = (
                    fallback_table.get("square_configs", {}).get(str(m))
                    if m == n == k
                    else fallback_table.get("rectangular_configs", {}).get(
                        f"{m}x{n}x{k}"
                    )
                )
            if previous_entry:
                configs = [dict(previous_entry[b]) for b in ("triton", "cuda")]
                configs += [
                    dict(backend="cuda", config_id=i)
                    for i, cfg in enumerate(cuda.configs_for_device())
                    if cfg.get("mma") == "tcgen05_ws"
                    and m % cfg["block_m"] == n % cfg.get("native_n_multiple", cfg["block_n"]) == 0
                    and i != previous_entry["cuda"].get("config_id")
                ]
            else:
                configs = list(TRITON_CONFIGS)
                configs += [
                    dict(backend="cuda", config_id=i)
                    for i, cfg in enumerate(cuda.configs_for_device())
                    if m % cfg["block_m"] == n % cfg.get("native_n_multiple", cfg["block_n"]) == 0
                ]
            if cutlass.is_available():
                configs += [
                    dict(backend="cutlass", config_id=i)
                    for i in (7, 15, 23, 31, 32, 33)
                    if i >= 32
                    or (
                        m % cutlass.CONFIGS[i]["block_m"]
                        == n % cutlass.CONFIGS[i]["block_n"]
                        == 0
                    )
                ]
            for cfg in configs:
                if cfg["backend"] == "triton":
                    functions.append(
                        lambda cfg=cfg: launch(a, b, sa, sb, out, 256, cfg)
                    )
                else:
                    module = cuda if cfg["backend"] == "cuda" else cutlass
                    functions.append(
                        lambda cfg=cfg, module=module: module.launch(
                            a, b, sa, sb, out, cfg["config_id"]
                        )
                    )
        else:
            for backend in ("auto", "cuda", "triton", "cutlass"):
                if backend == "cutlass" and not cutlass.is_available():
                    continue
                configs.append(
                    dict(
                        requested_backend=backend,
                        **get_kernel_config(
                            a, b, scale_a=sa, scale_b=sb, backend=backend
                        ),
                    )
                )
                functions.append(
                    lambda backend=backend: int8_gemm(a, b, sa, sb, backend=backend)
                )
            configs.append(dict(backend="torch_bf16"))
            functions.append(lambda: torch.mm(x, w.T))
            # Save the original untuned portable implementation for an honest before/after.
            configs.append(
                dict(
                    backend="portable_triton",
                    **{k: v for k, v in TRITON_CONFIGS[0].items() if k != "backend"},
                )
            )
            functions.append(lambda: launch(a, b, sa, sb, out, 256, TRITON_CONFIGS[0]))
        for cfg, fn in zip(configs, functions):
            result = fn()
            if result is None or not isinstance(result, torch.Tensor):
                result = out
            if cfg["backend"] == "torch_bf16":
                checks.append(dict(finite=torch.isfinite(result).all().item()))
            else:
                checks.append(check_sample(a, b, sa, sb, result, 256))
                if not args.tune and cfg.get("requested_backend") in (
                    "auto",
                    "cuda",
                    "cutlass",
                ):
                    checks[-1]["graph_replay"] = check_graph_replay(
                        fn, a, result, torch
                    )
            for _ in range(10):
                fn()
            start = time.monotonic()
            while (time.monotonic() - start) * 1000 < args.warmup_ms:
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        samples = [[] for _ in configs]
        for rnd in range(args.rounds):
            for step in range(len(configs)):
                index = (step + rnd) % len(configs)
                samples[index].append(
                    do_bench_cudagraph(
                        functions[index], rep=args.rep_ms, return_mode="median"
                    )
                )
        rows = [
            dict(
                config=c,
                round_medians_ms=s,
                median_ms=statistics.median(s),
                **{
                    (
                        "effective_tflops"
                        if c["backend"] == "torch_bf16"
                        else "effective_tops"
                    ): 2
                    * m
                    * n
                    * k
                    / (statistics.median(s) * 1e9)
                },
                correctness=ck,
            )
            for c, s, ck in zip(configs, samples, checks)
        ]
        case = dict(
            shape=[m, n, k],
            reused_fallback_candidates=bool(args.tune and previous_entry),
            results=rows,
            input_strides={
                name: list(t.stride())
                for name, t in (("a", a), ("b", b), ("scale_a", sa), ("scale_b", sb))
            },
        )
        if args.tune:
            best_by = {
                backend: min(
                    (r for r in rows if r["config"]["backend"] == backend),
                    key=lambda r: r["median_ms"],
                )
                for backend in ("triton", "cuda", "cutlass")
                if any(r["config"]["backend"] == backend for r in rows)
            }
            # Auto remains conservative: native needs a measured >=5% advantage.
            best = (
                best_by["cuda"]
                if best_by["cuda"]["median_ms"] < best_by["triton"]["median_ms"] / 1.05
                else best_by["triton"]
            )
            fallback = best
            if (
                "cutlass" in best_by
                and best_by["cutlass"]["median_ms"]
                < min(best_by["triton"]["median_ms"], best_by["cuda"]["median_ms"])
                / 1.05
            ):
                best = best_by["cutlass"]
            entry = {b: r["config"] for b, r in best_by.items()}
            entry["fallback_selected"] = fallback["config"]
            entry["selected"] = best["config"]
            if m == n == k:
                dispatch["square_configs"][str(m)] = entry
            else:
                dispatch["rectangular_configs"][f"{m}x{n}x{k}"] = entry
            case["selected"] = best["config"]
            print(
                f'{m}x{n}x{k}: {best["median_ms"]*1000:.2f} us {best["config"]}',
                flush=True,
            )
        else:
            print(
                f"{m}x{n}x{k}: "
                + " | ".join(
                    f'{r["config"].get("requested_backend",r["config"]["backend"])} {r["median_ms"]*1000:.2f}us'
                    for r in rows
                ),
                flush=True,
            )
        report["cases"].append(case)
        save(output, report)
        del functions, a, b, sa, sb, x, w, weights, scales, out
    report["complete"] = True
    save(output, report)
    if args.tune:
        dispatch["tuning_report_sha256"] = hashlib.sha256(
            output.read_bytes()
        ).hexdigest()
        if args.dispatch_output.exists():
            previous = json.loads(args.dispatch_output.read_text())
            if previous.get("compute_capability") == [11, 0]:
                for key in ("square_configs", "rectangular_configs"):
                    dispatch[key] = {**previous.get(key, {}), **dispatch[key]}
        save(args.dispatch_output, dispatch)


if __name__ == "__main__":
    main()
