"""Run the Thor square comparison one size at a time, cooling between sizes.

All measurement arguments and defaults come from compare_thor.py. Additional
options: --resume, --cooldown-temperature 60, --requested-gpu-clock-mhz 1200.
This runner records the existing clock setting; it never changes clocks/power.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import compare_thor as comparison
from plot_thor import load_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(comparison.__file__).resolve()
HASH_FIELDS = (
    "comparison_script_sha256", "shared_comparison_script_sha256",
    "device_check_sha256", "check_sample_script_sha256",
    "native_library_sha256", "native_build", "package_source_sha256",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cooldown-temperature", type=float, default=60)
    parser.add_argument("--requested-gpu-clock-mhz", type=float)
    extra, remaining = parser.parse_known_args(argv)
    if not math.isfinite(extra.cooldown_temperature) or extra.cooldown_temperature < 0:
        parser.error("cooldown-temperature must be finite and nonnegative")
    if extra.requested_gpu_clock_mhz is not None and (
        not math.isfinite(extra.requested_gpu_clock_mhz) or extra.requested_gpu_clock_mhz <= 0
    ):
        parser.error("requested-gpu-clock-mhz must be finite and positive")
    return extra, comparison.parse_args(remaining)


def validate_report(report, args, expected_sizes):
    if (report.get("expected_sizes") != expected_sizes
            or report.get("group_size") != args.group_size
            or report.get("seed") != args.seed
            or report.get("compute_capability") != [11, 0]):
        raise ValueError("report sizes, group size, seed or Thor architecture do not match")
    for key, expected in (("rounds", args.rounds), ("rep_ms", args.rep_ms),
                          ("warmup_calls_per_implementation", args.warmup)):
        if report.get("timing", {}).get(key) != expected:
            raise ValueError(f"report timing.{key} does not match")
    cases = report.get("reports")
    if not isinstance(cases, list):
        raise ValueError("reports must be a list of completed cases")
    sizes = [case.get("shape", {}).get("m") for case in cases]
    if any(size not in expected_sizes for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("completed cases contain an unexpected or duplicate size")
    if cases:
        # Apply the existing full numerical/correctness validator to the
        # completed subset. A failure marker is never a completed result.
        snapshot = dict(report, complete=True, expected_sizes=sizes)
        with tempfile.TemporaryDirectory(prefix="thor-validate-") as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            load_report(path)
    return sizes


def check_current_sources(report):
    source = report["source"]
    scripts = {
        "comparison_script_sha256": SCRIPT,
        "shared_comparison_script_sha256": SCRIPT.with_name("compare_backends.py"),
        "device_check_sha256": SCRIPT,
        "check_sample_script_sha256": SCRIPT.with_name("bench_sglang.py"),
        "native_library_sha256": ROOT / "src/grain_gemm/kernels/_native/libgrain_cuda.so",
    }
    for key, path in scripts.items():
        if source.get(key) != digest(path):
            raise ValueError(f"cannot resume: current {key} differs")
    for relative, expected in source["package_source_sha256"].items():
        if digest(ROOT / "src" / relative) != expected:
            raise ValueError(f"cannot resume: package source differs: {relative}")
    manifest = ROOT / "src/grain_gemm/kernels/_native/build.json"
    if json.loads(manifest.read_text()) != source["native_build"]:
        raise ValueError("cannot resume: native build metadata differs")


def cool_down(temperature):
    next_message = 0
    while True:
        state = comparison.gpu_state()
        if (state["temperature_c"] <= temperature
                and state["thermal_slowdown"].strip().lower() == "not active"):
            return state
        if time.monotonic() >= next_message:
            print(f"Cooling GPU: {state['temperature_c']:.2f} C; "
                  f"thermal slowdown {state['thermal_slowdown']}", flush=True)
            next_message = time.monotonic() + 10
        time.sleep(2)


def main(argv=None):
    extra, args = parse_args(argv)
    aggregate = None
    completed = []
    if extra.resume:
        aggregate = json.loads(args.output.read_text(encoding="utf-8"))
        completed = validate_report(aggregate, args, args.sizes)
        check_current_sources(aggregate)
        if completed and not aggregate.get("measurement_segments"):
            aggregate["measurement_segments"] = [{
                "kind": "resumed_existing", "sizes": list(completed),
                "started_at_utc": aggregate["started_at_utc"],
                "completed_at_utc": aggregate["reports"][-1].get(
                    "gpu_state_after", {}).get("timestamp_utc", aggregate.get("updated_at_utc")),
            }]
    for size in args.sizes:
        if size in completed:
            continue
        state = cool_down(extra.cooldown_temperature)
        with tempfile.TemporaryDirectory(prefix="thor-case-") as directory:
            output = Path(directory) / "case.json"
            command = [sys.executable, str(SCRIPT), "--sizes", str(size),
                       "--seed", str(args.seed), "--rounds", str(args.rounds),
                       "--rep-ms", str(args.rep_ms), "--warmup", str(args.warmup),
                       "--group-size", str(args.group_size), "--output", str(output)]
            print(f"Measuring M=N=K={size} after cooling to {state['temperature_c']:.2f} C", flush=True)
            subprocess.run(command, check=True)
            child = json.loads(output.read_text(encoding="utf-8"))
        if child.get("complete") is not True or validate_report(child, args, [size]) != [size]:
            raise ValueError("child comparison did not complete the requested size")
        if aggregate is None:
            aggregate = dict(child, expected_sizes=args.sizes, reports=[], measurement_segments=[])
        else:
            for key in ("gpu", "compute_capability", "software", "gpu_properties", "timing"):
                if aggregate[key] != child[key]:
                    raise ValueError(f"cannot merge: {key} differs between measurements")
            for key in HASH_FIELDS:
                if aggregate["source"].get(key) != child["source"].get(key):
                    raise ValueError(f"cannot merge: source.{key} differs")
        aggregate["reports"].extend(child["reports"])
        aggregate["measurement_segments"].append({
            "kind": "isolated_case", "sizes": [size],
            "started_at_utc": child["started_at_utc"],
            "completed_at_utc": child["completed_at_utc"],
        })
        aggregate["complete"] = False
        aggregate.pop("failure", None)
        aggregate.pop("completed_at_utc", None)
        aggregate["source"]["sweep_runner_script_sha256"] = digest(Path(__file__))
        applied = state["gpc_devfreq_hz"].get("cur_freq", state["graphics_clock_mhz"] * 1_000_000)
        requested = extra.requested_gpu_clock_mhz
        if requested is not None:
            aggregate["frequency_setting"] = {
                "requested_hz": round(requested * 1_000_000),
                "applied_hz": applied,
                "note": f"GPU clock reduced to {requested / 1000:g} GHz; actual supported setting {applied / 1_000_000:g} MHz",
            }
        else:
            aggregate.pop("frequency_setting", None)
            aggregate["recorded_clock_hz"] = applied
        aggregate["gpu_state_after"] = child["gpu_state_after"]
        comparison.shared.export(aggregate, args.output)
        completed.append(size)
    validate_report(aggregate, args, args.sizes)
    aggregate["complete"] = True
    aggregate.pop("failure", None)
    aggregate["completed_at_utc"] = comparison.shared.utc_now()
    aggregate["gpu_state_after"] = comparison.gpu_state()
    comparison.shared.export(aggregate, args.output)
    print(f"Saved all {len(completed)} sizes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
