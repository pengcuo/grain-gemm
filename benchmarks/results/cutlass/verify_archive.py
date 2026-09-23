#!/usr/bin/env python3
"""Verify published measurements using only Python's standard library (no GPU)."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


def check_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write an optional audit JSON")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    for item in manifest["files"]:
        data = (root / item["path"]).read_bytes()
        assert len(data) == item["bytes"], item["path"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]

    counts = Counter()
    paired_cases = {}
    optimized_cases = {}
    paths = list((root / "optimized/results").glob("case_*.json"))
    paths += list((root / "residual-gap").glob("paired_*/case_*.json"))
    for path in paths:
        case = json.loads(path.read_text())
        dims = case["shape"]
        shape = tuple(dims[x] for x in ("m", "n", "k"))
        ops = 2 * math.prod(shape)
        assert case["group_size"] == 256
        assert case["output_dtype"] == "bfloat16"
        is_paired = "orders" in case
        if is_paired:
            paired_cases[(path.parent.name, shape)] = case
            counts["paired_cases"] += 1
            assert len(case["orders"]) == case["rounds"] == 60
            variants = set(case["results"])
            assert all(set(order) == variants and len(order) == len(variants)
                       for order in case["orders"])
            for variant in variants:
                positions = Counter(order.index(variant) for order in case["orders"])
                assert max(positions.values()) - min(positions.values()) <= 1
            records = list(case["results"].items())
        else:
            optimized_cases[shape] = case
            counts["optimized_cases"] += 1
            counts["tuning_records"] += len(case["tuning"])
            records = list(case["results"].items())
            records += [("tuning", entry) for entry in case["tuning"]]

        for name, record in records:
            rounds = record["rounds_us"]
            assert all(all(math.isfinite(x) and x > 0 for x in row) for row in rounds)
            medians = [statistics.median(row) for row in rounds]
            assert medians == record["round_medians_us"]
            med = statistics.median(medians)
            check_close(med, record["median_us"])
            assert record["correctness"]["max_abs_error"] == 0.0
            if is_paired:
                assert len(rounds) == 60 and all(len(row) == 3 for row in rounds)
                ref = case["results"]["cute"]["round_medians_us"]
                check_close(statistics.median(a / b for a, b in zip(medians, ref)),
                            record["paired_ratio_to_cute"])
                contrast = case["contrasts"][name]
                ref = case["results"][contrast["reference"]]["round_medians_us"]
                check_close(statistics.median(a / b for a, b in zip(medians, ref)),
                            contrast["median_ratio"])
                counts["paired_paths"] += 1
                counts["paired_replay_samples"] += sum(map(len, rounds))
            else:
                assert len(rounds) == (3 if name == "tuning" else 5)
                assert all(len(row) == (3 if name == "tuning" else 5) for row in rounds)
                check_close(record["effective_tops"], ops / (med * 1e6))
                counts["optimized_replay_samples"] += sum(map(len, rounds))
                if name != "tuning":
                    counts["optimized_final_records"] += 1

    assert counts["optimized_cases"] == 18
    assert counts["tuning_records"] == 1188
    assert counts["optimized_final_records"] == 144
    assert counts["paired_cases"] == 19 and counts["paired_paths"] == 115
    assert counts["paired_replay_samples"] == 20700

    with (root / "optimized/results/results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    for row in rows:
        shape = tuple(int(row[k]) for k in ("m", "n", "k"))
        case = optimized_cases[shape]
        for name, record in case["results"].items():
            check_close(float(row[name + "_us"]), record["median_us"])
            check_close(float(row[name + "_equivalent_tflops"]), record["effective_tops"])
            assert json.loads(row[name + "_config"]) == record["config"]
            counts["csv_rows_checked"] += 1
    for file, suite in [("results_run3.csv", "paired_run3"),
                        ("paired_m4096/results.csv", "paired_m4096")]:
        with (root / "residual-gap" / file).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == (15 if suite == "paired_run3" else 20)
        for row in rows:
            shape = tuple(int(row[k]) for k in ("M", "N", "K"))
            case = paired_cases[(suite, shape)]
            record = case["results"][row["variant"]]
            check_close(float(row["median_us"]), record["median_us"])
            throughput = float(row.get("TOPS", row.get("equivalent_TFLOPS")))
            check_close(throughput, 2 * math.prod(shape) / (record["median_us"] * 1e6))
            check_close(float(row["paired_latency_change_vs_cute_pct"]),
                        (record["paired_ratio_to_cute"] - 1) * 100)
            if "source_sha256" in row:
                source = root / "residual-gap" / suite / Path(row["source_json"]).name
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
            counts["csv_rows_checked"] += 1

    # This verifier owns only these publication pages; other result stages have separate audits.
    pages = [root / "README.md"]
    pages += list((root / "optimized").rglob("*.md"))
    pages += list((root / "residual-gap").rglob("*.md"))
    for page in pages:
        for href in re.findall(r"\[[^\]]*\]\(([^)]+)\)", page.read_text()):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", href) or href.startswith("#"):
                continue
            href = unquote(href.split("#", 1)[0])
            assert not href.startswith("/"), (page, href)
            assert (page.parent / href).exists(), (page, href)
            counts["relative_links_checked"] += 1

    audit = {
        "status": "PASS",
        "verbatim_and_derived_file_hashes_checked": len(manifest["files"]),
        **dict(counts),
        "limitations": [
            "CPU-only audit of published bytes and stored timings; no GPU or binary revalidation.",
            "Historical binary/source audits are archived evidence, not rerun here.",
            "Bootstrap intervals are retained; this verifier recomputes point estimates, not bootstrap draws.",
            "Sampled correctness records are checked; original inputs are not retained for independent recomputation.",
        ],
    }
    text = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
