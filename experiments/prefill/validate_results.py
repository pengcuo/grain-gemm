#!/usr/bin/env python3
"""Validate and render the local LLM prefill G256 benchmark results.

This script is CPU-only. It reads results.json and writes a table-only report.md
beside the input, unless --output-dir is supplied. Pass --plots to additionally
render throughput curves and a speedup heatmap as PNG/SVG figures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


OPERATIONS = (
    ("q", "dense", "Query", 3072, 1536),
    ("k_or_v", "dense", "Key or value", 512, 1536),
    ("qkv", "dense", "Fused QKV", 4096, 1536),
    ("attention_out", "dense", "Attention output", 1536, 3072),
    ("shared_gate_or_up", "dense", "Shared gate or up", 1024, 1536),
    ("shared_gate_up", "dense", "Shared fused gate + up", 2048, 1536),
    ("shared_down", "dense", "Shared down", 1536, 1024),
    ("router", "dense", "Router", 32, 1536),
    ("expert_gate_or_up", "expert", "Expert gate or up", 1024, 1536),
    ("expert_gate_up", "expert", "Expert fused gate + up", 2048, 1536),
    ("expert_down", "expert", "Expert down", 1536, 1024),
    ("lm_head_151936", "head_all", "LM head · vocab 151936", 151936, 1536),
    ("lm_head_262144", "head_all", "LM head · vocab 262144", 262144, 1536),
    ("lm_head_last_151936", "head_last", "Last-token head · vocab 151936", 151936, 1536),
    ("lm_head_last_262144", "head_last", "Last-token head · vocab 262144", 262144, 1536),
)
BACKENDS = ("triton_default", "cute_default", "triton_tuned", "cute_tuned")
DEFAULT_DEPS = Path(__file__).resolve().with_name(".deps")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_number(value, description):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{description} must be a number")
    require(math.isfinite(value) and value > 0, f"{description} must be finite and positive")
    return float(value)


def close(actual, expected, description):
    actual = positive_number(actual, description)
    require(math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-9),
            f"{description}: recorded {actual:g}, recomputed {expected:g}")


def check_correctness_flags(value, location):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"passed", "pass", "ok", "valid", "success", "allclose", "within_tolerance",
                               "graph_original_zero_restore"}:
                require(item is not False, f"Failed correctness flag: {location}.{key}")
            if key.lower() == "status" and isinstance(item, str):
                require(item.lower() not in {"fail", "failed", "error"},
                        f"Failed correctness status: {location}.{key}={item}")
            check_correctness_flags(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_correctness_flags(item, f"{location}[{index}]")


def validate(data):
    """Return (operation, token count) records after consistency checks."""
    require(data.get("complete") is True, "Benchmark must be marked complete=true")
    require(isinstance(data.get("gpu"), str) and data["gpu"].strip(), "Missing GPU description")
    tokens = data.get("tokens")
    require(isinstance(tokens, list) and tokens, "Missing token sweep")
    require(all(type(t) is int and t > 0 and t % 8 == 0 for t in tokens),
            "Dense token counts must be positive integers divisible by eight")
    require(tokens == sorted(set(tokens)), "Token sweep must be sorted and unique")
    timing = data.get("timing", {})
    for key in ("rounds", "rep_ms", "tune_rounds", "tune_rep_ms"):
        positive_number(timing.get(key), f"timing.{key}")
    require(type(timing["rounds"]) is int, "timing.rounds must be an integer")
    require(type(timing["tune_rounds"]) is int, "timing.tune_rounds must be an integer")
    operations = {item[0]: item for item in OPERATIONS}
    cases = data.get("cases")
    require(isinstance(cases, list) and cases, "Missing benchmark cases")
    records = {}
    seen_shapes = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        shape = case.get("shape", {})
        require(all(type(shape.get(d)) is int and shape[d] > 0 for d in ("m", "n", "k")),
                f"{prefix}: invalid shape")
        m, n, k = (shape[d] for d in ("m", "n", "k"))
        require((m, n, k) not in seen_shapes, f"Duplicate shape {(m, n, k)}; combine its uses")
        seen_shapes.add((m, n, k))
        native_eligible = m % 64 == 0 and n % 64 == 0 and k % 256 == 0
        if "native_supported" in case:
            require(case["native_supported"] is native_eligible, f"{prefix}: native eligibility mismatch")
        if "group_size" in case:
            require(case["group_size"] == 256, f"{prefix}: report requires G256")
        if "output_dtype" in case:
            require(str(case["output_dtype"]).lower() in {"bf16", "bfloat16", "torch.bfloat16"},
                    f"{prefix}: report requires BF16 output")
        results, configs = case.get("results", {}), case.get("configs", {})
        for backend in BACKENDS:
            stats = results.get(backend)
            if backend.startswith("cute") and not native_eligible:
                require(stats is None, f"{prefix}.{backend}: native CUDA is unsupported for this shape")
                require(configs.get(backend) is None, f"{prefix}.{backend}: unsupported shape has a config")
                continue
            require(isinstance(stats, dict), f"{prefix}: missing results.{backend}")
            require(isinstance(configs.get(backend), dict), f"{prefix}: missing configs.{backend}")
            median = positive_number(stats.get("median_ms"), f"{prefix}.{backend}.median_ms")
            minimum = positive_number(stats.get("min_ms"), f"{prefix}.{backend}.min_ms")
            maximum = positive_number(stats.get("max_ms"), f"{prefix}.{backend}.max_ms")
            require(minimum <= median <= maximum, f"{prefix}.{backend}: min/median/max are out of order")
            rounds = stats.get("round_medians_ms")
            require(isinstance(rounds, list) and len(rounds) == timing["rounds"],
                    f"{prefix}.{backend}: round count does not match timing metadata")
            for round_index, value in enumerate(rounds):
                positive_number(value, f"{prefix}.{backend}.round[{round_index}]")
            close(median, statistics.median(rounds), f"{prefix}.{backend}.median_ms")
            close(minimum, min(rounds), f"{prefix}.{backend}.min_ms")
            close(maximum, max(rounds), f"{prefix}.{backend}.max_ms")
            close(stats.get("effective_tops"), 2 * m * n * k / (median * 1e9),
                  f"{prefix}.{backend}.effective_tops")
        for variant in ("default", "tuned"):
            reported = case.get(f"{variant}_speedup")
            if native_eligible:
                expected = results[f"triton_{variant}"]["median_ms"] / results[f"cute_{variant}"]["median_ms"]
                close(reported, expected, f"{prefix}.{variant}_speedup")
            else:
                require(reported is None, f"{prefix}.{variant}_speedup must be null for unsupported CuTe")
        correctness = case.get("correctness")
        require(isinstance(correctness, dict) and correctness, f"{prefix}: missing correctness metadata")
        check_correctness_flags(correctness, f"{prefix}.correctness")
        require(isinstance(case.get("tuning"), list), f"{prefix}: missing tuning records")
        uses = case.get("uses")
        require(isinstance(uses, list) and uses, f"{prefix}: missing workload uses")
        for use in uses:
            operation_id = use.get("id")
            require(operation_id in operations, f"{prefix}: unknown operation {operation_id!r}")
            _, category, _, expected_n, expected_k = operations[operation_id]
            require(use.get("category") == category, f"{prefix}.{operation_id}: category mismatch")
            require(isinstance(use.get("label"), str), f"{prefix}.{operation_id}: missing label")
            t = use.get("tokens")
            expected_tokens = [1] if category == "head_last" else tokens
            require(type(t) is int and t in expected_tokens, f"{prefix}.{operation_id}: invalid tokens {t}")
            expected_m = t // 8 if category == "expert" else t
            require((m, n, k) == (expected_m, expected_n, expected_k),
                    f"{prefix}.{operation_id}: shape {(m, n, k)} does not match expected "
                    f"{(expected_m, expected_n, expected_k)}")
            key = operation_id, t
            require(key not in records, f"Duplicate workload {key}")
            records[key] = {"case": case, "use": use, "native_eligible": native_eligible}
    expected = {(operation_id, t) for operation_id, category, *_ in OPERATIONS
                for t in ([1] if category == "head_last" else tokens)}
    require(set(records) == expected, f"Missing workloads: {sorted(expected - set(records))}")
    return records


def import_plotting(deps):
    if deps and deps.exists():
        sys.path.insert(0, str(deps))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    return plt, np, Normalize, Rectangle


def make_figure(data, records, deps):
    plt, np, Normalize, Rectangle = import_plotting(deps)
    columns = [*data["tokens"], 1]
    ratios = [record["case"][f"{variant}_speedup"] for record in records.values()
              for variant in ("default", "tuned") if record["native_eligible"]]
    extent = max(1.0, math.ceil(max(abs(math.log2(value)) for value in ratios)))
    normalizer = Normalize(vmin=-extent, vmax=extent)
    cmap = plt.get_cmap("RdBu")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 10.8), sharey=True)
    fig.subplots_adjust(left=0.225, right=0.925, bottom=0.16, top=0.85, wspace=0.10)
    for ax, variant, title in zip(axes, ("default", "tuned"),
                                  ("Current forced-backend defaults", "Locally tuned configurations")):
        values = np.full((len(OPERATIONS), len(columns)), np.nan)
        for row, (operation_id, category, *_rest) in enumerate(OPERATIONS):
            for col, t in enumerate(columns):
                record = records.get((operation_id, t))
                if record is not None and record["native_eligible"]:
                    values[row, col] = math.log2(record["case"][f"{variant}_speedup"])
        ax.imshow(values, cmap=cmap, norm=normalizer, aspect="auto", interpolation="nearest")
        for row, (operation_id, category, *_rest) in enumerate(OPERATIONS):
            for col, t in enumerate(columns):
                record = records.get((operation_id, t))
                if record is None:
                    ax.add_patch(Rectangle((col - .5, row - .5), 1, 1,
                                           facecolor="#f4f5f7", edgecolor="white", linewidth=1))
                    label, color = "—", "#a4a9b1"
                elif not record["native_eligible"]:
                    ax.add_patch(Rectangle((col - .5, row - .5), 1, 1,
                                           facecolor="#d6dce4", edgecolor="white", linewidth=1))
                    label, color = "N/A", "#354355"
                else:
                    value = record["case"][f"{variant}_speedup"]
                    label = f"{value:.2f}×" if value >= .01 else f"{value:.3f}×"
                    rgba = cmap(normalizer(math.log2(value)))
                    luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    color = "white" if luminance < .51 else "#17212f"
                ax.text(col, row, label, ha="center", va="center", color=color, fontsize=10)
        ax.set_title(title, fontsize=14, pad=14)
        ax.set_xticks(range(len(columns)), [*[f"{t:,}" for t in data["tokens"]], "Last token\nM = 1"])
        ax.set_yticks(range(len(OPERATIONS)), [item[2] for item in OPERATIONS])
        ax.set_xlabel("Prompt tokens T (expert M = T / 8)", labelpad=12)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(OPERATIONS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in (7.5, 10.5, 12.5):
            ax.axhline(boundary, color="#778497", linewidth=1.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([.944, .235, .013, .535])
    ticks = list(range(-int(extent), int(extent) + 1))
    if len(ticks) > 9:
        ticks = sorted(set([-int(extent), -2, -1, 0, 1, 2, int(extent)]))
    mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colorbar = fig.colorbar(mappable, cax=color_ax, ticks=ticks)
    colorbar.ax.set_yticklabels([f"{2 ** tick:g}×" for tick in ticks])
    colorbar.set_label("Triton time / CuTe time (log scale)", labelpad=10)
    gpu = data["gpu"].replace("\n", " ")
    fig.suptitle("LLM linear GEMMs · INT8 G256 · BF16 output", x=.50, y=.965,
                 fontsize=20, fontweight="bold")
    fig.text(.50, .921, f"{gpu}  |  Blue: CuTe faster (>1×) · Red: Triton faster (<1×)",
             ha="center", fontsize=12, color="#344154")
    timing = data["timing"]
    footnote = (
        f"Final timing: median of {timing['rounds']} round medians; rep = {timing['rep_ms']:g} ms. "
        f"Tuning: {timing['tune_rounds']} rounds; rep = {timing['tune_rep_ms']:g} ms.\n"
        "N/A: native CuTe unsupported (M or N is not a multiple of 64). —: workload does not use this token column.\n"
        "Separate and fused projections are alternatives. Expert rows measure one expert; all-token LM heads represent optional prompt logprobs.\n"
        "Default/tuned repeats of the same configuration are retimed; small differences can reflect measurement noise.\n"
        "Synthetic GEMMs only; routing, attention QK/PV, softmax, quantization, and end-to-end model execution are outside these timings."
    )
    fig.text(.225, .037, footnote, fontsize=9.7, color="#344154", linespacing=1.55)
    return fig, plt


def make_curve_figure(data, records, deps, variant):
    """Plot throughput versus dense token count for each supported operation."""
    require(variant in {"default", "tuned"}, "Unknown curve configuration variant")
    plt, np, _Normalize, _Rectangle = import_plotting(deps)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.titleweight": "bold"})
    operations = [item for item in OPERATIONS if item[0] != "router" and item[1] != "head_last"]
    require(len(operations) == 12, "Curve layout expects 12 native-supported operations")
    tokens = np.asarray(data["tokens"], dtype=float)
    fig, axes = plt.subplots(4, 3, figsize=(18.5, 17.5))
    fig.subplots_adjust(left=.066, right=.98, bottom=.115, top=.862, wspace=.25, hspace=.65)
    for ax, (operation_id, category, label, n, k) in zip(axes.flat, operations):
        ceiling = 0.0
        for backend, backend_label, color, marker in (
            ("triton", "Triton", "#2166ac", "o"),
            ("cute", "CuTe", "#d95f02", "s"),
        ):
            median_tops, minimum_tops, maximum_tops = [], [], []
            for t in data["tokens"]:
                record = records[operation_id, t]
                require(record["native_eligible"], f"Curves require native support: {operation_id}, T={t}")
                case = record["case"]
                stats = case["results"][f"{backend}_{variant}"]
                work = 2 * case["shape"]["m"] * n * k / 1e9
                median_tops.append(stats["effective_tops"])
                minimum_tops.append(work / stats["max_ms"])
                maximum_tops.append(work / stats["min_ms"])
            ax.fill_between(tokens, minimum_tops, maximum_tops,
                            color=color, alpha=.16, linewidth=0, zorder=1)
            ax.plot(tokens, median_tops, color=color, marker=marker, linewidth=2.1,
                    markersize=6.5, label=backend_label, zorder=2)
            ceiling = max(ceiling, *maximum_tops)
        occupancy = "M = T / 8" if category == "expert" else "M = T"
        subtitle = f"N = {n:,} · K = {k:,} · {occupancy}"
        if category == "head_all":
            label += " (optional prompt logprobs)"
        ax.set_title(f"{label}\n{subtitle}", fontsize=10.7, linespacing=1.55, pad=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tokens, [str(t) for t in data["tokens"]])
        ax.minorticks_off()
        ax.set_xlim(tokens[0] / 1.09, tokens[-1] * 1.09)
        ax.set_ylim(0, ceiling * 1.13)
        ax.set_xlabel("Total prompt tokens T", labelpad=5)
        ax.set_ylabel("Effective INT8 TOPS", labelpad=5)
        ax.grid(axis="both", color="#dbe1e8", linewidth=.7, zorder=0)
        ax.tick_params(axis="both", labelsize=9.5, colors="#344154")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#a8b1bd")
        ax.spines["left"].set_color("#a8b1bd")
    variant_label = ("Current forced-backend defaults" if variant == "default"
                     else "Offline locally tuned configurations")
    fig.suptitle(f"LLM prefill GEMM throughput: {variant_label}", x=.51, y=.973,
                 fontsize=20, fontweight="bold")
    fig.text(.51, .945, f"{data['gpu']} · INT8 G256 · FP32 scales · BF16 output",
             ha="center", fontsize=12.5, color="#344154")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .93),
               ncol=2, frameon=False, fontsize=12, handlelength=3, columnspacing=3)
    timing = data["timing"]
    footnote = (
        f"Points: throughput from median latency over {timing['rounds']} timing rounds "
        f"(rep = {timing['rep_ms']:g} ms). Shading: throughput range across round medians, not a confidence interval.\n"
        "Each panel has its own TOPS scale. Dense/shared/head M = T; one routed expert M = T / 8 assumes balanced routing. "
        "Separate and fused projections are alternatives.\n"
        "All-token LM heads are optional prompt-logprob workloads. Router (N = 32) and last-token heads (M = 1) "
        "lack native CuTe support and appear in the supplemental heatmap and tables.\n"
        "Synthetic preallocated-output GEMMs only; no end-to-end/model-accuracy claim. Current auto dispatch uses Triton here. "
        "Repeated identical default/tuned configs can differ because of timing noise."
    )
    fig.text(.066, .03, footnote, fontsize=9.5, color="#344154", linespacing=1.65)
    return fig, plt


def latency_cell(stats):
    if stats is None:
        return "N/A"
    return f"{1000 * stats['median_ms']:,.3f}"


def ratio_cell(value):
    return "N/A" if value is None else f"{value:.3f}×"


def throughput_pair(triton_stats, cute_stats):
    def value(stats):
        return "N/A" if stats is None else f"{stats['effective_tops']:.2f}"
    return f"{value(triton_stats)} / {value(cute_stats)}"


def report_text(data, records, input_path, include_plots=False):
    lines = [
        "# LLM prefill GEMM benchmark: INT8 G256",
        "",
        f"GPU: **{data['gpu']}**. Source: [{input_path.name}]({input_path.as_posix()}). "
        f"{len(data['cases'])} unique matrix shapes cover {len(records)} workload uses.",
        "",
        "Speedup means **Triton median time / CuTe median time**. Values above 1 favor CuTe; "
        "values below 1 favor Triton. Current forced-backend defaults and locally tuned "
        "configurations are compared separately; default CuTe timings do not describe automatic dispatch.",
        "",
        "## Timing and validation",
        "",
        "The tables use medians of the recorded round medians. Each backend's recorded "
        "minimum and maximum, round count, effective TOPS, and both speedup ratios were checked. "
        "Workload coverage, projection dimensions, per-expert occupancy, and native-CUDA eligibility "
        "were also checked before rendering. Effective TOPS is `2 × M × N × K / (median_ms × 10^9)`.",
        "",
        "Timing metadata from the benchmark:",
        "",
        "```json",
        json.dumps(data["timing"], indent=2, ensure_ascii=False),
        "```",
        "",
        "All per-round measurements, sampled correctness metadata, tuning candidates, and selected "
        "configuration dictionaries remain available in the source JSON.",
        "",
        "Default and tuned selections are timed separately, including when they select the same "
        "configuration. Small differences between those repeated measurements do not demonstrate "
        "a tuning improvement; inspect the selected configurations and round variability before "
        "attributing a change to tuning. Resource-limited non-default tuning candidates may be skipped; "
        "the candidate search is local and does not establish a globally optimal configuration.",
        "",
        "## Workload interpretation",
        "",
        "- The model uses hidden size 1536, 24 layers, 24 query heads and 4 KV heads with head "
        "dimension 128. Query width is therefore 3072; each key/value width is 512.",
        "- FFNs assume gated SiLU (SwiGLU): two 1536→1024 gate/up projections and one "
        "1024→1536 down projection. There are 32 routed experts, top-4 routing, and one shared expert.",
        "- Dense/shared projections use M=T. A routed expert uses M=T×4/32=T/8 under balanced "
        "routing. Its row measures one ordinary GEMM, not grouped-MoE performance or end-to-end "
        "expert dispatch and combination. Real occupancy imbalance is not represented.",
        "- Q versus fused QKV, and separate gate/up versus fused gate+up, are implementation "
        "alternatives. Their rows must not be added together. The K-or-V and gate-or-up rows "
        "represent one projection each.",
        "- Attention QK/PV matrix multiplications, softmax, activations, routing, and communication "
        "are outside this linear-projection benchmark.",
        "- All-token LM heads (M=T) represent optional prompt-logprob or full-logit workloads. "
        "Typical single-sequence generation after prefill needs only the last-token head (M=1), "
        "which is unsupported by this native CuTe kernel and is shown separately.",
        "- Inputs are synthetic INT8 tensors with G256 quantization groups, FP32 scales, and BF16 "
        "outputs. Quantization cost is excluded. These measurements establish neither model "
        "accuracy nor end-to-end inference latency.",
        "- Native CuTe requires SM121, aligned contiguous A and B.T, contiguous scales, M/N "
        "multiples of 64, and K a multiple of 256. N=32 router and M=1 last-token head cases "
        "are N/A; Triton still runs them.",
        "- The current automatic dispatch selects Triton for these rectangular LLM shapes. "
        "CuTe measurements use the explicit native backend. This experiment changes neither "
        "kernel code nor dispatch policy; local tuning only selects benchmark configurations.",
        "",
    ]
    if data.get("resume_sessions"):
        insert_at = lines.index("## Timing and validation") + 2
        lines[insert_at:insert_at] = [
            "Sweep resumed after host restart; completed cases preserved, remaining shapes "
            "measured with unchanged kernel binaries/software/timing settings. The sweep therefore "
            "spans multiple sessions rather than one uninterrupted run.",
            "",
        ]
    csv_path = input_path.with_suffix(".csv")
    tables = [
        "## Measurements by total token count",
        "",
        "Backend columns show median latency in **µs**. Every row lists M, N, and K explicitly. "
        "T is the total dense prompt-token count; routed-expert M=T/8 assumes balanced routing. "
        "N/A means native CuTe does not support the shape.",
        "",
        "The added **Equivalent TFLOPS (Triton / CuTe)** column uses "
        "`2 × M × N × K / (latency_us × 10^6)`, computed from unrounded timings. "
        "These are INT8 GEMMs: the physical unit is **TOPS**, and the equivalent-TFLOPS "
        "label is an operation-count convention with the same numerical value, not a "
        "measurement of floating-point arithmetic throughput.",
        "",
        f"Effective TOPS and detailed measurements are retained in [{csv_path.name}]({csv_path.as_posix()}) "
        "and the source JSON.",
        "",
    ]
    for t in [*data["tokens"], 1]:
        title = f"Total prompt tokens T = {t:,}" if t != 1 else "Last-token LM heads (M = 1)"
        tables += [f"### {title}", ""]
        for variant, variant_title in (
            ("default", "Current forced-backend defaults"),
            ("tuned", "Offline locally tuned configurations"),
        ):
            tables += [f"**{variant_title}**", "",
                       "| Operator | M | N | K | Triton µs | CuTe µs | Triton / CuTe | Equivalent TFLOPS (Triton / CuTe) |",
                       "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for operation_id, _category, fallback_label, *_ in OPERATIONS:
                record = records.get((operation_id, t))
                if record is None:
                    continue
                case = record["case"]
                label = (record["use"]["label"] or fallback_label).replace("|", "\\|").replace("\n", " ")
                shape, stats = case["shape"], case["results"]
                cells = [f"{label} (`{operation_id}`)",
                         *[f"{shape[dimension]:,}" for dimension in ("m", "n", "k")],
                         latency_cell(stats[f"triton_{variant}"]), latency_cell(stats[f"cute_{variant}"]),
                         ratio_cell(case[f"{variant}_speedup"]),
                         throughput_pair(stats[f"triton_{variant}"], stats[f"cute_{variant}"])]
                tables.append("| " + " | ".join(cells) + " |")
            tables.append("")
    if include_plots:
        tables += [
            "## Optional throughput curves",
            "",
            "Each panel plots effective INT8 TOPS against total prompt tokens T on a base-2 logarithmic "
            "axis. Shading spans throughput derived from minimum/maximum round-median latency, "
            "not a confidence interval. Each panel has its own TOPS scale.",
            "",
            "![Current forced-backend default throughput curves](llm_prefill_curves_default.png)",
            "",
            "[Default curves: vector figure](llm_prefill_curves_default.svg)",
            "",
            "![Offline locally tuned throughput curves](llm_prefill_curves_tuned.png)",
            "",
            "[Locally tuned curves: vector figure](llm_prefill_curves_tuned.svg)",
            "",
            "## Optional speedup heatmap",
            "",
            "![Default and locally tuned CuTe/Triton comparison](llm_prefill_speedup.png)",
            "",
            "[Vector figure](llm_prefill_speedup.svg)",
            "",
        ]
    insertion = lines.index("## Timing and validation")
    lines[insertion:insertion] = tables
    lines += ["## Shape reuse", "",
              "Identical [M,N,K] shapes are timed once and reused for every matching workload. "
              "This includes shared and routed-expert projections where their token counts happen "
              "to produce the same GEMM shape.", ""]
    reused = [case for case in data["cases"] if len(case["uses"]) > 1]
    if reused:
        lines += ["| [M, N, K] | Workload uses |", "| :--- | :--- |"]
        for case in sorted(reused, key=lambda item: tuple(item["shape"][d] for d in ("m", "n", "k"))):
            shape = ", ".join(f"{case['shape'][d]:,}" for d in ("m", "n", "k"))
            uses = "; ".join(f"`{use['id']}` T={use['tokens']:,}" for use in case["uses"])
            lines.append(f"| [{shape}] | {uses} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().with_name("results.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--plots", action="store_true",
                        help="Opt in to PNG/SVG curves, a heatmap, and image embeds in report.md")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    input_path = args.input.resolve()
    data = json.loads(input_path.read_text())
    records = validate(data)
    print(f"Validated {len(data['cases'])} unique shapes and {len(records)} workload uses.")
    if args.validate_only:
        return
    output_dir = (args.output_dir or input_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = report_text(data, records, input_path, include_plots=args.plots)
    output = output_dir / "report.md"
    output.write_text(report)
    print(output)
    if not args.plots:
        return
    for variant in ("default", "tuned"):
        fig, plt = make_curve_figure(data, records, args.plot_deps, variant)
        for extension in ("png", "svg"):
            output = output_dir / f"llm_prefill_curves_{variant}.{extension}"
            fig.savefig(output, dpi=190, facecolor="white")
            print(output)
        plt.close(fig)
    fig, plt = make_figure(data, records, args.plot_deps)
    for extension in ("png", "svg"):
        output = output_dir / f"llm_prefill_speedup.{extension}"
        fig.savefig(output, dpi=190, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
