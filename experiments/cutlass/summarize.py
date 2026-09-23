#!/usr/bin/env python3
"""Generate a CSV and Markdown table with TOPS from paired case JSON (CPU only)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing case_*.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh summary directory")
    args = parser.parse_args()
    paths = sorted(args.input.expanduser().resolve().glob("case_*.json"))
    if not paths:
        parser.error("No case_*.json files found")
    output = args.output.expanduser().resolve()
    if (output / "results.csv").exists() or (output / "table.md").exists():
        parser.error("Refusing to overwrite an existing results.csv or table.md")
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        m, n, k = (data["shape"][key] for key in ("m", "n", "k"))
        for name, result in data["results"].items():
            latency = statistics.median(statistics.median(r) for r in result["rounds_us"])
            if abs(latency - result["median_us"]) > 1e-9:
                raise ValueError(f"Stored median mismatch in {path.name}: {name}")
            rows.append(dict(M=m, N=n, K=k, G=data["group_size"], variant=name,
                             config=result["config_id"], median_us=latency,
                             TOPS=2*m*n*k/(latency*1e6),
                             paired_latency_change_vs_cute_pct=100*(result["paired_ratio_to_cute"]-1),
                             source_json=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows.sort(key=lambda row: (row["M"], row["N"], row["K"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["| M | N | K | Implementation | Latency μs ↓ | TOPS ↑ |",
             "|---:|---:|---:|---|---:|---:|"]
    table += [f"| {r['M']} | {r['N']} | {r['K']} | {r['variant']} | {r['median_us']:.2f} | {r['TOPS']:.2f} |"
              for r in rows]
    (output / "table.md").write_text("\n".join(table) + "\n\nTOPS = 2*M*N*K / (latency_us * 10^6).\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
