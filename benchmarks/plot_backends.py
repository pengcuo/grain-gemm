"""Plot forced CuTe and Triton INT8 versus PyTorch BF16 across all 16 square sizes."""

import argparse
import json
import math
from pathlib import Path
import statistics


METHODS = ("cute_int8_gemm", "triton_int8_gemm", "torch_mm_bf16")
EXPECTED_SIZES = list(range(1024, 16385, 1024))


def positive_number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be a finite positive number")
    return value


def positive_integer(value, description):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def matching_number(actual, expected, description):
    positive_number(actual, description)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{description} is inconsistent with the round measurements")


def load_report(path):
    """Validate completeness, numerical summaries, and the displayed methodology."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("complete") is not True:
        raise ValueError("the benchmark sweep must have complete=true")
    reports = data.get("reports")
    expected_sizes = data.get("expected_sizes")
    if not isinstance(reports, list) or not reports:
        raise ValueError("the benchmark must contain at least one completed case")
    if not isinstance(expected_sizes, list) or not expected_sizes:
        raise ValueError("expected_sizes must be a nonempty list")
    for size in expected_sizes:
        positive_integer(size, "expected size")
    if sorted(expected_sizes) != EXPECTED_SIZES:
        raise ValueError("this plot requires all 16 sizes: 1024, 2048, ..., 16384")
    if data.get("forced_backends") != {"cute_int8_gemm": "cuda", "triton_int8_gemm": "triton"}:
        raise ValueError("this plot requires explicit CUDA/CuTe and Triton backend measurements")
    if not isinstance(data.get("gpu"), str) or not data["gpu"].strip():
        raise ValueError("GPU name is required")
    group_size = data.get("group_size")
    if type(group_size) is not int or group_size != 256:
        raise ValueError("expected group size 256")
    timing = data["timing"]
    rounds = positive_integer(timing["rounds"], "timing rounds")
    positive_number(timing["rep_ms"], "target timing batch duration")
    if timing["method"] != "triton.testing.do_bench_cudagraph":
        raise ValueError("this plot requires CUDA Graph measurements")
    if timing["cache_policy"] != "repeated inputs; no cache flush":
        raise ValueError("this plot requires repeated inputs without cache flushing")
    for field in (
        "includes_input_quantization", "includes_compilation_or_autotuning",
        "includes_warmup", "includes_host_dispatch",
    ):
        if timing.get(field) is not False:
            raise ValueError(f"timing.{field} must be false for this GEMM-only plot")
    if data.get("quantization", {}).get("performed_before_timing") is not True:
        raise ValueError("input quantization must be performed before timing")
    for field in ("torch", "triton", "cuda"):
        if not isinstance(data["software"].get(field), str) or not data["software"][field]:
            raise ValueError(f"software.{field} is required")

    sizes = []
    for case in reports:
        shape = case["shape"]
        m, n, k = (positive_integer(shape[axis], f"shape.{axis}") for axis in ("m", "n", "k"))
        if m != n or m != k:
            raise ValueError(f"expected square M=N=K, got {shape}")
        sizes.append(m)
        if case["group_size"] != group_size or k % group_size:
            raise ValueError(f"inconsistent group size or incomplete quantization groups at {shape}")
        if case["output_dtype"] != "bfloat16 for all implementations":
            raise ValueError("all three implementations must produce BF16 output")
        if case["layout"] != {"a": "row-major", "b": "column-major", "scales": "row-major"}:
            raise ValueError(f"unexpected input layout at {shape}")
        for name, backend in data["forced_backends"].items():
            config = case.get("kernel_configs", {}).get(name)
            if not isinstance(config, dict) or config.get("backend") != backend:
                raise ValueError(f"missing forced {backend} kernel configuration at {shape}")
            replay = case.get("graph_replay_checks", {}).get(name, {})
            for check in ("original_matches_eager", "zeroed_activation_gives_zero",
                          "restored_activation_matches_eager"):
                if replay.get(check) is not True:
                    raise ValueError(f"missing CUDA Graph replay check {check}: {name}, {shape}")

        checks = case.get("sampled_correctness", {})
        for name in METHODS[:2]:
            check = checks.get(name)
            if not isinstance(check, dict):
                raise ValueError(f"missing independent INT32-reference check: {name}, {shape}")
            for axis in ("rows", "columns"):
                count = positive_integer(check[f"checked_{axis}"], f"{name}: checked {axis}")
                if count != min(32, m):
                    raise ValueError(f"unexpected correctness sample count: {name}, {shape}")
            error = check["max_abs_error"]
            if isinstance(error, bool) or not isinstance(error, (int, float)):
                raise ValueError(f"invalid correctness error: {name}, {shape}")
            if not math.isfinite(error) or error < 0:
                raise ValueError(f"invalid correctness error: {name}, {shape}")

        orders = case["measurement_order"]
        if not isinstance(orders, list) or len(orders) != rounds:
            raise ValueError(f"missing measurement orders at {shape}")
        if any(not isinstance(order, list) or sorted(order) != sorted(METHODS) for order in orders):
            raise ValueError(f"each round must time all three implementations at {shape}")
        for name in METHODS:
            summary = case["results"][name]
            samples = summary["round_medians_ms"]
            if not isinstance(samples, list) or len(samples) != rounds:
                raise ValueError(f"expected {rounds} round medians: {name}, {shape}")
            for sample in samples:
                positive_number(sample, f"round latency: {name}, {shape}")
            expected = {
                "median_ms": statistics.median(samples),
                "min_ms": min(samples), "max_ms": max(samples),
            }
            throughput_key = "effective_tflops" if name == "torch_mm_bf16" else "effective_tops"
            expected[throughput_key] = 2 * m * n * k / (expected["median_ms"] * 1e9)
            for field, value in expected.items():
                matching_number(summary[field], value, f"{field}: {name}, {shape}")

        medians = {name: case["results"][name]["median_ms"] for name in METHODS}
        ratios = {
            "cute_speedup_over_bf16": medians["torch_mm_bf16"] / medians["cute_int8_gemm"],
            "cute_speedup_over_triton": medians["triton_int8_gemm"] / medians["cute_int8_gemm"],
            "triton_speedup_over_bf16": medians["torch_mm_bf16"] / medians["triton_int8_gemm"],
        }
        for field, value in ratios.items():
            matching_number(case[field], value, f"{field}: {shape}")
    if sorted(sizes) != sorted(expected_sizes):
        raise ValueError("completed cases must match expected_sizes exactly, without duplicates")
    data["reports"] = sorted(reports, key=lambda case: case["shape"]["m"])
    return data


def plot_report(data, output, svg=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 15, "axes.labelsize": 12,
        "axes.edgecolor": "#BBC4D0", "axes.labelcolor": "#243247",
        "xtick.color": "#44546A", "ytick.color": "#44546A",
        "text.color": "#142438", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    reports = data["reports"]
    sizes = [case["shape"]["m"] for case in reports]
    scale_x = all(size % 1024 == 0 for size in sizes)
    x_values = [size / 1024 if scale_x else size for size in sizes]
    x_label = "Matrix size M = N = K (×1024)" if scale_x else "Matrix size M = N = K"
    colors = {"cute_int8_gemm": "#0072B2", "triton_int8_gemm": "#D55E00", "torch_mm_bf16": "#009E73"}
    markers = {"cute_int8_gemm": "o", "triton_int8_gemm": "s", "torch_mm_bf16": "^"}
    labels = {
        "cute_int8_gemm": "GrainGEMM CUDA / CuTe INT8 (TOPS)",
        "triton_int8_gemm": "GrainGEMM Triton INT8 (TOPS)",
        "torch_mm_bf16": "PyTorch BF16 (TFLOPS)",
    }
    fig, (throughput_ax, speedup_ax) = plt.subplots(
        2, 1, figsize=(16, 10.5), gridspec_kw={"height_ratios": [1.16, 1]},
    )
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.225, top=0.82, hspace=0.51)
    fig.suptitle(
        f"GrainGEMM G{data['group_size']}: CuTe vs. Triton vs. BF16",
        x=0.085, y=0.962, ha="left", fontsize=25, fontweight="bold",
    )
    size_range = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}"
    fig.text(
        0.085, 0.916,
        f"{data['gpu']}  |  M = N = K: {size_range}  |  "
        f"{len(sizes)} measured sizes  |  Prequantized INT8 inputs  |  BF16 output",
        fontsize=12.5, color="#44546A",
    )

    for name in METHODS:
        summaries = [case["results"][name] for case in reports]
        operations = [2 * size**3 for size in sizes]
        throughput_ax.fill_between(
            x_values,
            [ops / (entry["max_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            [ops / (entry["min_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            color=colors[name], alpha=0.12, linewidth=0,
        )
        throughput_ax.plot(
            x_values,
            [ops / (entry["median_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            color=colors[name], marker=markers[name], markersize=5.5,
            linewidth=2.3, label=labels[name],
        )
    throughput_ax.set_ylim(bottom=0)
    throughput_ax.set_ylabel("Effective throughput\n(TOPS / TFLOPS)")
    throughput_ax.set_title("Throughput · higher is better", loc="left", pad=32)
    throughput_ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3,
        frameon=False, fontsize=10.2, borderaxespad=0, columnspacing=2.0,
    )

    ratios = (
        ("cute_speedup_over_bf16", "CuTe vs. PyTorch BF16", "#0072B2", "o"),
        ("cute_speedup_over_triton", "CuTe vs. GrainGEMM Triton", "#D55E00", "s"),
    )
    all_ratios = []
    for key, label, color, marker in ratios:
        values = [case[key] for case in reports]
        all_ratios.extend(values)
        speedup_ax.plot(x_values, values, color=color, marker=marker,
                        markersize=5.5, linewidth=2.3, label=label)
    speedup_ax.axhline(1, color="#65758B", linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    lower, upper = min(1, min(all_ratios)), max(1, max(all_ratios))
    padding = max(0.08, (upper - lower) * 0.14)
    speedup_ax.set_ylim(max(0, lower - padding), upper + padding)
    speedup_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}×"))
    speedup_ax.set_ylabel("Comparison latency /\nCuTe latency")
    speedup_ax.set_title("CuTe speedup · above 1× is faster", loc="left", pad=32)
    speedup_ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2,
        frameon=False, fontsize=10.2, borderaxespad=0, columnspacing=2.0,
    )
    speedup_ax.annotate(
        "1× = equal latency", xy=(1, 1), xycoords=("axes fraction", "data"),
        xytext=(-5, 5), textcoords="offset points", ha="right", va="bottom",
        fontsize=9.5, color="#526278",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    for axis in (throughput_ax, speedup_ax):
        span = x_values[-1] - x_values[0]
        margin = max(span * 0.025, 0.25 if scale_x else max(1, x_values[0] * 0.04))
        axis.set_xlim(x_values[0] - margin, x_values[-1] + margin)
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{value:g}" for value in x_values], rotation=0 if len(sizes) <= 20 else 45)
        axis.set_xlabel(x_label, labelpad=8)
        axis.tick_params(axis="both", length=0, pad=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DCE2E9", linewidth=0.8)
        axis.set_axisbelow(True)

    timing, software = data["timing"], data["software"]
    footer = (
        f"Timing: median of {timing['rounds']} round medians with CUDA Graphs; "
        f"target captured batch {timing['rep_ms']:g} ms. Shading: round min–max, not confidence intervals.",
        "Each point is one measured square size. Throughput = 2MNK / time: INT8 uses integer TOPS; BF16 uses floating-point TFLOPS.",
        "GEMM includes group scaling, accumulation, and output conversion. Quantization, compilation, autotuning, and warmup are excluded.",
        "Forced backends; repeated inputs; no cache flush. Per-size kernel settings, graph checks, and raw rounds are in the JSON.",
        f"PyTorch {software['torch']}  |  Triton {software['triton']}  |  CUDA {software['cuda']}  |  GrainGEMM · measured results",
    )
    for index, text in enumerate(footer):
        fig.text(0.085, 0.143 - index * 0.026, text, fontsize=10 if index < 4 else 9,
                 color="#44546A" if index < 4 else "#65758B")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    if svg:
        svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg)
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed compare_backends.py JSON aggregate for all 16 sizes")
    parser.add_argument("--output", type=Path, required=True, help="output PNG file")
    parser.add_argument("--svg", type=Path, help="optional companion SVG file")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("output must have a .png suffix")
    if args.svg and args.svg.suffix.lower() != ".svg":
        parser.error("svg must have a .svg suffix")
    try:
        data = load_report(args.input)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid benchmark report: {error}")
    plot_report(data, args.output, args.svg)
    print(f"Plotted {len(data['reports'])} measured sizes: {args.output}")


if __name__ == "__main__":
    main()
