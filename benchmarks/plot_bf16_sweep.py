"""Plot the measured GB10 G256 square GEMM sweep; requires matplotlib."""

import argparse
import json
import math
from pathlib import Path
import statistics


METHODS = ("sglang_int8_gemm", "torch_mm_bf16")
SIZES = list(range(1024, 16385, 1024))


def load_reports(path):
    """Reject incomplete sweeps or inconsistent benchmark summaries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("complete") is False:
        raise ValueError("the benchmark sweep has not completed")
    reports = data["reports"]
    reports = sorted(reports, key=lambda report: report["shape"]["m"])
    if [report["shape"]["m"] for report in reports] != SIZES:
        raise ValueError("expected all 16 sizes from 1024 to 16384, exactly once")

    first = reports[0]
    shared_fields = (
        "gpu", "software", "compute_capability", "group_size", "output_dtype",
        "int8_config", "int8_config_autotuned", "layout",
        "torch_allow_bf16_reduced_precision_reduction",
    )
    shared_timing_fields = ("method", "statistic", "rounds", "rep_ms", "cache_policy")
    for report in reports:
        shape = report["shape"]
        if shape["m"] != shape["n"] or shape["m"] != shape["k"]:
            raise ValueError(f"expected a square GEMM, got {shape}")
        if report["group_size"] != 256 or "GB10" not in report["gpu"]:
            raise ValueError("this figure describes GB10 with group size 256")
        if report["int8_config_autotuned"]:
            raise ValueError("this figure describes the untuned baseline")
        if report["int8_config"] != {
            "block_m": 32, "block_n": 64, "num_warps": 4, "num_stages": 2,
        }:
            raise ValueError("expected the untuned 32 x 64, 4-warp, 2-stage config")
        if report["timing"]["rounds"] != 5:
            raise ValueError("expected five timing rounds per size")
        if report["timing"]["method"] != "triton.testing.do_bench_cudagraph":
            raise ValueError("expected CUDA Graph timing")
        if report["timing"]["cache_policy"] != "repeated inputs; no cache flush":
            raise ValueError("expected repeated inputs without cache flushing")
        if report["output_dtype"] != "bfloat16 for both implementations":
            raise ValueError("expected BF16 output from both implementations")
        if any(report[field] != first[field] for field in shared_fields):
            raise ValueError("GPU, software, layout, or kernel configuration differs")
        if any(
            report["timing"][field] != first["timing"][field]
            for field in shared_timing_fields
        ):
            raise ValueError("timing methodology differs between sizes")

        for name in METHODS:
            summary = report["results"][name]
            samples = summary["round_medians_ms"]
            if len(samples) != 5 or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"invalid round measurements at {shape}: {name}")
            expected = {
                "median_ms": statistics.median(samples),
                "min_ms": min(samples), "max_ms": max(samples),
            }
            for key, value in expected.items():
                if not math.isclose(summary[key], value, rel_tol=1e-9):
                    raise ValueError(f"inconsistent {key} at {shape}: {name}")
        ratio = (
            report["results"]["torch_mm_bf16"]["median_ms"]
            / report["results"]["sglang_int8_gemm"]["median_ms"]
        )
        if not math.isclose(report["int8_speedup_over_bf16"], ratio, rel_tol=1e-9):
            raise ValueError(f"inconsistent speedup at {shape}")
    return reports


def plot_reports(reports, output, svg=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": "#BBC4D0",
        "axes.labelcolor": "#243247",
        "xtick.color": "#44546A",
        "ytick.color": "#44546A",
        "text.color": "#142438",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, (throughput_ax, speedup_ax) = plt.subplots(1, 2, figsize=(16, 8.7))
    fig.subplots_adjust(left=0.075, right=0.973, bottom=0.29, top=0.79, wspace=0.24)
    fig.suptitle(
        "G256 INT8 GEMM vs. PyTorch BF16",
        x=0.075, y=0.955, ha="left", fontsize=25, fontweight="bold",
    )
    fig.text(
        0.075, 0.895,
        "NVIDIA GB10  |  M = N = K: 1024 to 16384  |  G = 256  |  BF16 output",
        fontsize=13, color="#44546A",
    )

    x_values = [report["shape"]["m"] // 1024 for report in reports]
    colors = {"sglang_int8_gemm": "#0072B2", "torch_mm_bf16": "#D55E00"}
    labels = {
        "sglang_int8_gemm": "INT8 G256 / SGLang Triton (TOPS)",
        "torch_mm_bf16": "PyTorch BF16 (TFLOPS)",
    }
    for name in METHODS:
        summaries = [report["results"][name] for report in reports]
        operations = [2 * report["shape"]["m"] ** 3 for report in reports]
        throughput_ax.fill_between(
            x_values,
            [ops / (entry["max_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            [ops / (entry["min_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            color=colors[name], alpha=0.14, linewidth=0,
        )
        throughput_ax.plot(
            x_values,
            [ops / (entry["median_ms"] * 1e9) for ops, entry in zip(operations, summaries)],
            color=colors[name], marker="o", markersize=5, linewidth=2.3,
            label=labels[name],
        )
    throughput_ax.set_ylim(bottom=0)
    throughput_ax.set_ylabel("Effective throughput (TOPS / TFLOPS)")
    throughput_ax.set_title("Throughput  ·  higher is better", loc="left", pad=18)
    throughput_ax.legend(loc="lower right", frameon=False, fontsize=10)

    speedups = [report["int8_speedup_over_bf16"] for report in reports]
    speedup_ax.axhline(1, color="#65758B", linewidth=1.4, linestyle=(0, (4, 3)))
    speedup_ax.plot(
        x_values, speedups, color="#7256A8", marker="o", markersize=5,
        linewidth=2.3,
    )
    speedup_ax.set_ylim(min(0.5, min(speedups) - 0.10), max(1.1, max(speedups) + 0.10))
    speedup_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}×"))
    speedup_ax.set_ylabel("BF16 latency / INT8 latency")
    speedup_ax.set_title("INT8 speedup  ·  higher is better", loc="left", pad=18)
    speedup_ax.text(
        15.9, 1, "1× = equal latency", ha="right", va="bottom", fontsize=10,
        color="#526278", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    speedup_ax.text(
        0.03, 0.04, "Above 1×: INT8 faster\nBelow 1×: BF16 faster",
        transform=speedup_ax.transAxes, fontsize=10, color="#526278",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    for axis in (throughput_ax, speedup_ax):
        axis.set_xlim(0.7, 16.3)
        axis.set_xticks(x_values)
        axis.set_xlabel("Matrix size M = N = K (×1024)", labelpad=12)
        axis.tick_params(axis="both", length=0, pad=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", which="major", color="#DCE2E9", linewidth=0.8)
        axis.set_axisbelow(True)

    timing = reports[0]["timing"]
    software = reports[0]["software"]
    fig.text(
        0.075, 0.165,
        "Timing: median of 5 round medians, measured with CUDA Graphs. "
        "Shading: min–max across rounds.",
        fontsize=10.5, color="#44546A",
    )
    fig.text(
        0.075, 0.13,
        "Throughput = 2MNK / time: INT8 uses integer TOPS; BF16 uses floating-point TFLOPS. "
        "Scaling work is included in timing.",
        fontsize=10.5, color="#44546A",
    )
    fig.text(
        0.075, 0.095,
        "GEMM execution only; quantization, compilation and warmup excluded. "
        "Repeated inputs; no cache flush.",
        fontsize=10.5, color="#44546A",
    )
    fig.text(
        0.075, 0.06,
        f"INT8 config: 32×64 tile, 4 warps, 2 stages, untuned. "
        f"Target graph batch: {timing['rep_ms']} ms; 10 replays per round.",
        fontsize=10.5, color="#44546A",
    )
    fig.text(
        0.075, 0.025,
        f"PyTorch {software['torch']}  |  Triton {software['triton']}  |  "
        f"CUDA {software['cuda']}  |  GrainGEMM · measured results",
        fontsize=9, color="#65758B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    if svg:
        svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="sweep JSON with a reports list")
    parser.add_argument("--output", type=Path, required=True, help="output PNG file")
    parser.add_argument("--svg", type=Path, help="optional companion SVG file")
    args = parser.parse_args()
    reports = load_reports(args.input)
    plot_reports(reports, args.output, args.svg)
    print(f"Plotted {len(reports)} measured sizes: {args.output}")


if __name__ == "__main__":
    main()
