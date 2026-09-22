"""Run independent square INT8/BF16 comparisons and export JSON plus CSV."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def export(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "m", "n", "k", "group_size", "int8_median_us", "bf16_median_us",
            "int8_min_us", "int8_max_us", "bf16_min_us", "bf16_max_us",
            "int8_effective_tops", "bf16_effective_tflops",
            "int8_speedup_over_bf16", "int8_latency_increase_percent",
        ])
        for case in report["reports"]:
            a = case["results"]["sglang_int8_gemm"]
            b = case["results"]["torch_mm_bf16"]
            writer.writerow([
                case["shape"]["m"], case["shape"]["n"], case["shape"]["k"],
                case["group_size"], a["median_ms"] * 1000, b["median_ms"] * 1000,
                a["min_ms"] * 1000, a["max_ms"] * 1000,
                b["min_ms"] * 1000, b["max_ms"] * 1000,
                a["effective_tops"], b["effective_tflops"],
                case["int8_speedup_over_bf16"],
                (a["median_ms"] / b["median_ms"] - 1) * 100,
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1024)
    parser.add_argument("--stop", type=int, default=16384)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128, 256], default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.start, args.stop, args.step, args.rounds, args.rep_ms) <= 0:
        parser.error("sizes, step, rounds, and rep-ms must be positive")
    if args.start > args.stop or (args.stop - args.start) % args.step:
        parser.error("stop must be reachable from start with the chosen step")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix; a matching .csv is also written")
    sizes = list(range(args.start, args.stop + 1, args.step))
    if any(size % args.group_size for size in sizes):
        parser.error("every size must be divisible by the quantization group size")

    comparison = Path(__file__).resolve().with_name("compare_bf16.py")
    report = {
        "benchmark": "SGLang INT8 versus PyTorch BF16 square GEMM sweep",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_sizes": sizes,
        "group_size": args.group_size,
        "comparison_script_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
        "complete": False,
        "reports": [],
    }
    with tempfile.TemporaryDirectory(prefix="grain-bf16-sweep-") as directory:
        work = Path(directory)
        for index, size in enumerate(sizes, 1):
            print(f"[{index}/{len(sizes)}] M=N=K={size}, G={args.group_size}", flush=True)
            result_path = work / f"{size}.json"
            log_path = work / f"{size}.log"
            command = [
                sys.executable, str(comparison), "--m", str(size), "--n", str(size),
                "--k", str(size), "--group-size", str(args.group_size),
                "--rounds", str(args.rounds), "--rep-ms", str(args.rep_ms),
                "--output", str(result_path),
            ]
            with log_path.open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                print(log_path.read_text(), file=sys.stderr)
                raise SystemExit(process.returncode)
            case = json.loads(result_path.read_text())
            report["reports"].append(case)
            export(report, args.output)
            a = case["results"]["sglang_int8_gemm"]["median_ms"]
            b = case["results"]["torch_mm_bf16"]["median_ms"]
            print(f"  INT8 {a:.6f} ms | BF16 {b:.6f} ms | ratio {b / a:.3f}x", flush=True)
    report["complete"] = True
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    export(report, args.output)
    print(f"Saved {len(sizes)} independent comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
