"""Plot completed Thor SM110 CuTe/Triton INT8 and PyTorch BF16 comparisons."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics

if __package__:
    from .plot_backends import (
        METHODS, matching_number, plot_report as plot_backend_report,
        positive_integer, positive_number,
    )
else:
    from plot_backends import (
        METHODS, matching_number, plot_report as plot_backend_report,
        positive_integer, positive_number,
    )


FORCED_BACKENDS = {"cute_int8_gemm": "cuda", "triton_int8_gemm": "triton"}


def require_mapping(value, description):
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def validate_frequency_setting(data):
    """Check recorded clock limits, without inferring clocks between snapshots."""
    if "frequency_setting" not in data:
        return
    setting = require_mapping(data["frequency_setting"], "frequency_setting")
    positive_integer(setting.get("requested_hz"), "frequency_setting.requested_hz")
    applied = positive_integer(setting.get("applied_hz"), "frequency_setting.applied_hz")
    snapshots = [("gpu_state_before", data.get("gpu_state_before"))]
    if "gpu_state_after" in data:
        snapshots.append(("gpu_state_after", data["gpu_state_after"]))
    for case in data["reports"]:
        for field in ("gpu_state_before", "gpu_state_after"):
            snapshots.append((f"{case['shape']}.{field}", case.get(field)))
    for description, snapshot in snapshots:
        state = require_mapping(snapshot, description)
        limits = require_mapping(state.get("gpc_devfreq_hz"), f"{description}.gpc_devfreq_hz")
        for field in ("min_freq", "max_freq"):
            value = positive_integer(limits.get(field), f"{description}.{field}")
            if value != applied:
                raise ValueError(f"{description}.{field} does not match frequency_setting.applied_hz")


def load_report(path):
    """Validate Thor identity, square shapes, and the shared timing methodology."""
    data = require_mapping(json.loads(path.read_text(encoding="utf-8")), "report")
    if data.get("complete") is not True:
        raise ValueError("the benchmark sweep must have complete=true")
    if (not isinstance(data.get("gpu"), str)
            or re.search(r"\b(?:Thor|T5000)\b", data["gpu"], re.IGNORECASE) is None):
        raise ValueError("this plot requires an NVIDIA Thor GPU report")
    capability = data.get("compute_capability")
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int for value in capability)
            or capability != [11, 0]):
        raise ValueError("this plot requires compute capability 11.0 (SM110)")
    if data.get("forced_backends") != FORCED_BACKENDS:
        raise ValueError("explicit CUDA/CuTe and Triton measurements are required")
    if type(data.get("group_size")) is not int or data["group_size"] != 256:
        raise ValueError("expected group size 256")

    expected_sizes = data.get("expected_sizes")
    if not isinstance(expected_sizes, list) or not expected_sizes:
        raise ValueError("expected_sizes must be a nonempty list")
    for size in expected_sizes:
        positive_integer(size, "expected size")
        if size % 256:
            raise ValueError("Thor sizes must be multiples of 256")
    if len(set(expected_sizes)) != len(expected_sizes):
        raise ValueError("expected_sizes must not contain duplicates")
    reports = data.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("the benchmark must contain at least one completed case")

    timing = require_mapping(data.get("timing"), "timing")
    rounds = positive_integer(timing["rounds"], "timing rounds")
    positive_number(timing["rep_ms"], "target timing batch duration")
    positive_integer(timing["warmup_calls_per_implementation"], "warmup calls")
    if timing.get("method") != "triton.testing.do_bench_cudagraph":
        raise ValueError("this plot requires CUDA Graph measurements")
    if timing.get("cache_policy") != "repeated inputs; no cache flush":
        raise ValueError("this plot requires repeated inputs without cache flushing")
    if timing.get("correctness_checks_performed_before_timing") is not True:
        raise ValueError("correctness checks must precede timing")
    for field in (
        "includes_input_quantization", "includes_compilation_or_autotuning",
        "includes_warmup", "includes_host_dispatch",
    ):
        if timing.get(field) is not False:
            raise ValueError(f"timing.{field} must be false for this GEMM-only plot")
    quantization = require_mapping(data.get("quantization"), "quantization")
    if quantization.get("performed_before_timing") is not True:
        raise ValueError("input quantization must be performed before timing")
    for field, expected in (
        ("input_dtype", "bfloat16"), ("quantized_dtype", "int8"),
        ("scale_dtype", "float32"),
    ):
        if quantization.get(field) != expected:
            raise ValueError(f"quantization.{field} must be {expected}")
    software = require_mapping(data.get("software"), "software")
    for field in ("torch", "triton", "cuda"):
        if not isinstance(software.get(field), str) or not software[field].strip():
            raise ValueError(f"software.{field} is required")

    sizes = []
    for case in reports:
        require_mapping(case, "case")
        shape = require_mapping(case.get("shape"), "shape")
        m, n, k = (positive_integer(shape[axis], f"shape.{axis}") for axis in ("m", "n", "k"))
        if m != n or m != k or m % 256:
            raise ValueError(f"expected square G256 shape, got {shape}")
        sizes.append(m)
        if type(case.get("group_size")) is not int or case["group_size"] != 256:
            raise ValueError(f"inconsistent group size at {shape}")
        if case.get("output_dtype") != "bfloat16 for all implementations":
            raise ValueError("all three implementations must produce BF16 output")
        if case.get("layout") != {"a": "row-major", "b": "column-major", "scales": "row-major"}:
            raise ValueError(f"unexpected input layout at {shape}")
        expected_strides = {
            "a": [k, 1], "b": [1, k],
            "scale_a": [k // 256, 1], "scale_b": [n, 1],
        }
        if case.get("input_strides") != expected_strides:
            raise ValueError(f"unexpected input or scale strides at {shape}")

        configs = require_mapping(case.get("kernel_configs"), "kernel_configs")
        graph_checks = require_mapping(case.get("graph_replay_checks"), "graph_replay_checks")
        correctness = require_mapping(case.get("sampled_correctness"), "sampled_correctness")
        for name, backend in FORCED_BACKENDS.items():
            config = require_mapping(configs.get(name), f"{name} kernel configuration")
            if config.get("backend") != backend:
                raise ValueError(f"missing forced {backend} kernel configuration at {shape}")
            replay = require_mapping(graph_checks.get(name), f"{name} graph checks")
            for check in (
                "original_matches_eager", "zeroed_activation_gives_zero",
                "restored_activation_matches_eager",
            ):
                if replay.get(check) is not True:
                    raise ValueError(f"missing CUDA Graph replay check {check}: {name}, {shape}")
            check = require_mapping(correctness.get(name), f"{name} independent correctness check")
            for axis in ("rows", "columns"):
                count = positive_integer(check[f"checked_{axis}"], f"{name}: checked {axis}")
                if count != min(32, m):
                    raise ValueError(f"unexpected correctness sample count: {name}, {shape}")
            error = check.get("max_abs_error")
            if (isinstance(error, bool) or not isinstance(error, (int, float))
                    or not math.isfinite(error) or error < 0):
                raise ValueError(f"invalid correctness error: {name}, {shape}")

        orders = case.get("measurement_order")
        if (not isinstance(orders, list) or len(orders) != rounds
                or any(not isinstance(order, list) or sorted(order) != sorted(METHODS) for order in orders)):
            raise ValueError(f"each of {rounds} rounds must time all three implementations at {shape}")
        results = require_mapping(case.get("results"), "results")
        medians = {}
        for name in METHODS:
            summary = require_mapping(results.get(name), f"{name} results")
            samples = summary.get("round_medians_ms")
            if not isinstance(samples, list) or len(samples) != rounds:
                raise ValueError(f"expected {rounds} round medians: {name}, {shape}")
            for sample in samples:
                positive_number(sample, f"round latency: {name}, {shape}")
            median = statistics.median(samples)
            expected = {"median_ms": median, "min_ms": min(samples), "max_ms": max(samples)}
            throughput_key = "effective_tflops" if name == "torch_mm_bf16" else "effective_tops"
            expected[throughput_key] = 2 * m * n * k / (median * 1e9)
            for field, value in expected.items():
                matching_number(summary[field], value, f"{field}: {name}, {shape}")
            medians[name] = median
        ratios = {
            "cute_speedup_over_bf16": medians["torch_mm_bf16"] / medians["cute_int8_gemm"],
            "cute_speedup_over_triton": medians["triton_int8_gemm"] / medians["cute_int8_gemm"],
            "triton_speedup_over_bf16": medians["torch_mm_bf16"] / medians["triton_int8_gemm"],
        }
        for field, value in ratios.items():
            matching_number(case[field], value, f"{field}: {shape}")
    if sorted(sizes) != sorted(expected_sizes):
        raise ValueError("completed cases must match expected_sizes exactly, without duplicates")
    validate_frequency_setting(data)
    data["reports"] = sorted(reports, key=lambda case: case["shape"]["m"])
    return data


def plot_report(data, output, svg=None):
    """Add frequency context to the figure while preserving the raw GPU name."""
    display_data = dict(data)
    setting = data.get("frequency_setting")
    clock_note = None
    if setting is not None:
        requested = positive_integer(setting.get("requested_hz"), "frequency_setting.requested_hz")
        applied = positive_integer(setting.get("applied_hz"), "frequency_setting.applied_hz")
        clock_note = f"GPU clock reduced to {requested / 1e9:g} GHz ({applied / 1e6:g} MHz)"
    else:
        state = data.get("gpu_state_before", {})
        if isinstance(state, dict):
            clocks = state.get("gpc_devfreq_hz", {})
            frequency = clocks.get("cur_freq") if isinstance(clocks, dict) else None
            if type(frequency) is int and frequency > 0:
                clock_note = f"GPU clock snapshot: {frequency / 1e6:g} MHz"
            else:
                frequency = state.get("graphics_clock_mhz")
                if type(frequency) is int and frequency > 0:
                    clock_note = f"GPU clock snapshot: {frequency:g} MHz"
    if clock_note:
        display_data["gpu"] = f"{data['gpu']}  |  {clock_note}"
    plot_backend_report(display_data, output, svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="completed compare_thor.py JSON; a size subset is accepted")
    parser.add_argument("--output", type=Path, required=True, help="output PNG file")
    parser.add_argument("--svg", type=Path, help="optional companion SVG file")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("output must have a .png suffix")
    if args.svg and args.svg.suffix.lower() != ".svg":
        parser.error("svg must have a .svg suffix")
    try:
        data = load_report(args.input)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid benchmark report: {error}")
    plot_report(data, args.output, args.svg)
    print(f"Plotted {len(data['reports'])} measured Thor sizes: {args.output}")


if __name__ == "__main__":
    main()
