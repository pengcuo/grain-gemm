#!/usr/bin/env python3
"""Read-only, standard-library audit of the M1024/M2048 comparison artifacts.

Does not import measurement modules, torch, or CUDA, and writes no files.
Exit status: 0 = complete/pass, 1 = audit failure, 2 = incomplete/no failures.
Correctness is audited from saved checks and the hashed measurement protocol;
this cannot independently recompute GEMMs because tensor inputs are not saved.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


BACKENDS = ("cutlass_before", "cutlass", "cuda", "triton",
            "cutlass_fp32_shared", "cutlass_bf16_shared",
            "cutlass_async_direct", "cutlass_reg_direct")
FAMILIES = dict(cutlass_fp32_shared=0, cutlass_bf16_shared=8,
                cutlass_async_direct=16, cutlass_reg_direct=24)
NK = ((1024, 1536), (1536, 1024), (2048, 1536), (3072, 1536),
      (512, 1536), (4096, 1536), (1536, 3072), (151936, 1536), (262144, 1536))
EXPECTED_SHAPES = {(m, n, k) for m in (1024, 2048) for n, k in NK}
TRITON_TILES = ((16, 32, 4, 2), (16, 64, 4, 2), (32, 64, 4, 2),
                (32, 128, 4, 3), (64, 64, 4, 2), (64, 128, 8, 3),
                (64, 128, 4, 3), (128, 64, 4, 3), (128, 128, 8, 2))
EXPECTED_CONFIGS = ([dict(backend="cutlass_before", config_id=i) for i in range(8)]
                    + [dict(backend="cutlass", config_id=i) for i in range(32)]
                    + [dict(backend="cuda", config_id=i) for i in range(17)]
                    + [dict(backend="triton", block_m=m, block_n=n,
                            num_warps=w, num_stages=s, swizzle=8)
                       for m, n, w, s in TRITON_TILES])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal assignment: {name}")


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.max_recorded_abs_error = 0.0
        self.raw_replay_samples = 0

    def require(self, condition, label):
        if not condition:
            self.errors.append(label)

    def number(self, actual, expected, label):
        self.require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                     and math.isfinite(actual)
                     and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                     f"{label}: saved={actual!r}, recomputed={expected!r}")

    def measurement(self, row, shape, rounds, replays, label):
        values = row["rounds_us"]
        self.require(len(values) == rounds and all(len(r) == replays for r in values),
                     f"{label}: wrong round/replay counts")
        self.require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                         and math.isfinite(x) and x > 0 for r in values for x in r),
                     f"{label}: invalid raw latency")
        medians = [statistics.median(r) for r in values]
        self.require(len(row["round_medians_us"]) == len(medians),
                     f"{label}: wrong number of saved round medians")
        for i, (saved, calculated) in enumerate(zip(row["round_medians_us"], medians)):
            self.number(saved, calculated, f"{label}: round {i} median")
        latency = statistics.median(medians)
        for key, expected in (("median_us", latency), ("min_round_us", min(medians)),
                              ("max_round_us", max(medians)),
                              ("effective_tops", 2 * math.prod(shape) / (latency * 1e6))):
            self.number(row[key], expected, f"{label}: {key}")
        nodes = row["graph_nodes"]
        self.require(type(nodes) is int and 1 <= nodes <= 256,
                     f"{label}: invalid graph node count")
        check = row["correctness"]
        self.require(check["checked_rows"] == min(32, shape[0])
                     and check["checked_columns"] == min(32, shape[1]),
                     f"{label}: incorrect sample dimensions")
        error = check["max_abs_error"]
        self.require(isinstance(error, (int, float)) and math.isfinite(error) and error >= 0,
                     f"{label}: invalid saved correctness error")
        self.max_recorded_abs_error = max(self.max_recorded_abs_error, error)
        self.raw_replay_samples += sum(map(len, values))


def audit_protocol(audit, source_text):
    """Inspect control-flow text from the hashed script, without executing it."""
    tree = ast.parse(source_text)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_case")
    statements = [ast.unparse(n) for n in run.body]
    expected_sequence = (
        "selected = {backend:", "torch.manual_seed(20260924)",
        "a.random_(-127, 128)", "b.T.random_(-127, 128)",
        "sa.uniform_(0.001, 0.01)", "sb.uniform_(0.001, 0.01)",
        "ref = reference(a, b, sa, sb, torch)",
        "for backend, i in selected.items():", "final = {backend:",
        "for round_id in range(5):", "results = {backend:",
    )
    cursor = 0
    for expected in expected_sequence:
        matches = [i for i in range(cursor, len(statements))
                   if statements[i].startswith(expected)]
        audit.require(bool(matches), f"protocol: missing/out-of-order {expected}")
        if matches:
            cursor = matches[0] + 1
    verification_loop = next((s for s in statements
                              if s.startswith("for backend, i in selected.items():")), "")
    for text in ("graph.replay()", "correctness[backend] = check(out, ref, torch)",
                 "a.zero_()", "assert torch.count_nonzero(out).item() == 0",
                 "a.copy_(saved_a)", "check(out, ref, torch)"):
        audit.require(text in verification_loop, f"protocol: missing fresh/zero/restore check: {text}")
    check_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    normalized_check = ast.unparse(check_function)
    audit.require("torch.testing.assert_close(actual, expected)" in normalized_check,
                  "protocol: missing sample correctness assertion")
    audit.require("assert torch.isfinite(out).all().item()" in normalized_check,
                  "protocol: missing full-output finite assertion")
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--package-dir", type=Path,
                        default=Path("/home/pengcuo/spark/grain-gemm/src/grain_gemm"))
    args = parser.parse_args()
    root = args.experiment_dir
    audit = Audit()
    scope = json.loads((root / "scope.json").read_text())
    shapes = [tuple(s) for s in scope["shapes"]]
    audit.require(len(shapes) == 18 and set(shapes) == EXPECTED_SHAPES,
                  "scope.json must contain the exact 18 model projection shapes, without duplicates")
    audit.require(scope["candidate_counts"] == dict(cutlass_before=8, cutlass=32, cute=17, triton=9),
                  "scope candidate counts differ from 8/32/17/9")
    audit.require(scope["group_size"] == 256 and scope["output"] == "bfloat16",
                  "scope is not G256/BF16")
    script = root / "compare_optimized.py"
    candidates_file = root / "tune_grain.py"
    tree = audit_protocol(audit, script.read_text())
    declared_shapes = [tuple(s) for s in literal_assignment(tree, "SHAPES")]
    audit.require(len(declared_shapes) == 18 and set(declared_shapes) == EXPECTED_SHAPES,
                  "measurement script SHAPES differs from scope")
    candidates = literal_assignment(ast.parse(candidates_file.read_text()), "TRITON_CANDIDATES")
    audit.require(candidates == EXPECTED_CONFIGS[57:], "nine Triton candidates differ from expected set")
    for source in (script, candidates_file):
        snapshot = root / "measurement_sources" / source.name
        if snapshot.is_file():
            audit.require(sha(snapshot) == sha(source), f"script snapshot mismatch: {source.name}")
        else:
            audit.warnings.append(f"script snapshot not present yet: {source.name}")

    files = sorted((root / "results").glob("case_*.json"))
    loaded = []
    seen = []
    measured_candidates = 0
    skipped_candidates = 0
    first = None
    expected_config_counter = Counter(map(canonical, EXPECTED_CONFIGS))
    expected_timing = dict(method="CUDA graphs", tuning_rounds=3, tuning_replays=3,
                           final_rounds=5, final_replays=5, max_graph_nodes=256,
                           target_graph_ms=5, reduction="median of round medians",
                           order="rotated each round", warm_cache=True,
                           preallocated_output=True,
                           scope="GEMM, group scaling and BF16 conversion; no quantization or host dispatch")
    for path in files:
        try:
            data = json.loads(path.read_text())
            shape = tuple(data["shape"][axis] for axis in ("m", "n", "k"))
            label = "x".join(map(str, shape))
            seen.append(shape)
            audit.require(shape in EXPECTED_SHAPES, f"unexpected shape: {shape}")
            audit.require(path.name == f"case_{'_'.join(map(str, shape))}.json",
                          f"case filename/shape mismatch: {path.name}")
            audit.require(data.get("complete") is True, f"{label}: not marked complete")
            audit.require(data["group_size"] == 256 and data["output_dtype"] == "bfloat16",
                          f"{label}: incorrect group/output type")
            audit.require(data["seeds"] == dict(tuning=20260923, final=20260924),
                          f"{label}: fresh-input seeds changed")
            audit.require(data["timing"] == expected_timing, f"{label}: unexpected timing protocol")
            env = data["environment"]
            audit.require(env["device"] == "NVIDIA GB10" and env["compute_capability"] == [12, 1]
                          and env["multiprocessors"] == 48, f"{label}: wrong device")
            audit.require(data["device"] == env["device"] and
                          data["compute_capability"] == env["compute_capability"] and
                          data["software"] == {key: env[key] for key in ("torch", "triton", "cuda")},
                          f"{label}: inconsistent software/device fields")
            if first is None:
                first = data
            for field in ("source", "environment", "software", "device", "compute_capability", "timing", "seeds"):
                audit.require(data[field] == first[field], f"{label}: {field} differs between cases")
            tune, skipped = data["tuning"], data["skipped"]
            attempted = Counter(canonical(row["config"]) for row in tune + skipped)
            audit.require(attempted == expected_config_counter,
                          f"{label}: candidate set must cover all 66 configurations exactly once")
            for row in skipped:
                audit.require(bool(row.get("reason")), f"{label}: skipped candidate lacks reason")
            if skipped:
                audit.warnings.append(f"{label}: {len(skipped)} candidates skipped; no measurements for these")
            for i, row in enumerate(tune):
                audit.measurement(row, shape, 3, 3, f"{label} tuning[{i}]")
            audit.require(set(data["results"]) == set(BACKENDS), f"{label}: missing final backend")
            for backend in BACKENDS:
                eligible = [row for row in tune if (
                    row["config"]["backend"] == backend if backend not in FAMILIES else
                    row["config"]["backend"] == "cutlass" and
                    FAMILIES[backend] <= row["config"]["config_id"] < FAMILIES[backend]+8)]
                best = min(eligible, key=lambda r: r["median_us"])
                final = data["results"][backend]
                audit.require(final["config"] == best["config"],
                              f"{label}: {backend} final config is not its independent tuning winner")
                audit.require(final["graph_nodes"] == best["graph_nodes"],
                              f"{label}: {backend} final graph differs from selected tuning graph")
                audit.measurement(final, shape, 5, 5, f"{label} final {backend}")
            measured_candidates += len(tune)
            skipped_candidates += len(skipped)
            loaded.append(data)
        except (KeyError, TypeError, ValueError, OSError, StopIteration, ZeroDivisionError) as exc:
            audit.errors.append(f"{path.name}: malformed/unreadable case: {exc}")

    audit.require(len(seen) == len(set(seen)), "duplicate shape records")
    if first:
        provenance = first["source"]
        baseline_manifest = root / "before_hashes.json"
        baseline = root / "before/src/grain_gemm/kernels/_native/libgrain_cutlass.so"
        audit.require(provenance["baseline_library_sha256"] == sha(baseline),
                      "baseline library differs from measured copy")
        audit.require(provenance["baseline_manifest_sha256"] == sha(baseline_manifest),
                      "baseline manifest hash differs")
        for relative, expected_hash in json.loads(baseline_manifest.read_text()).items():
            path = root / "before" / relative
            audit.require(path.is_file() and sha(path) == expected_hash,
                          f"baseline source/library snapshot mismatch: {relative}")
        audit.require(sha(baseline) == "60142480c47f31aa990f656e5309852edb98f94e29cb366a0329224a70b3bc51",
                      "baseline is not the original measured 8-config CUTLASS binary")
        audit.require(provenance["libraries"]["libgrain_cutlass.so"] != sha(baseline),
                      "optimized library is identical to baseline binary")
        audit.require(provenance["script_sha256"] == sha(script), "measurement script hash differs from recorded hash")
        audit.require(provenance["candidate_script_sha256"] == sha(candidates_file),
                      "candidate script hash differs from recorded hash")
        snapshot_package = root / "measurement_sources/grain_gemm"
        if not snapshot_package.is_dir():
            snapshot_package = root / "measurement_sources/src/grain_gemm"
        for relative, expected_hash in provenance["sources"].items():
            for label, base in (("snapshot", snapshot_package), ("current package", args.package_dir)):
                path = base / relative
                audit.require(path.is_file() and sha(path) == expected_hash,
                              f"{label} source hash mismatch: {relative}")
        snapshot_sources = {str(p.relative_to(snapshot_package)) for p in snapshot_package.rglob("*")
                            if p.is_file() and p.suffix in (".py", ".cu", ".hpp", ".json")}
        audit.require(snapshot_sources == set(provenance["sources"]),
                      "snapshot source inventory differs from recorded inventory")
        audit.require(set(provenance["libraries"]) == {"libgrain_cuda.so", "libgrain_cutlass.so"},
                      "native library inventory differs")
        for name, expected_hash in provenance["libraries"].items():
            path = args.package_dir / "kernels/_native" / name
            audit.require(path.is_file() and sha(path) == expected_hash,
                          f"native library hash mismatch: {name}")
        build = json.loads((snapshot_package / "kernels/_native/build.json").read_text())
        cutlass_build = json.loads((snapshot_package / "kernels/_native/cutlass_build.json").read_text())
        audit.require(build["source_sha256"] == sha(snapshot_package / "kernels/csrc/grain_cute.cu"),
                      "CuTe build manifest does not match measured source")
        for name, expected_hash in cutlass_build["source_sha256"].items():
            audit.require(sha(snapshot_package / "kernels/csrc" / name) == expected_hash,
                          f"CUTLASS build manifest mismatch: {name}")
        audit.require(build["architecture"] == cutlass_build["architecture"] == "sm_121",
                      "native build architecture mismatch")

    missing = sorted(EXPECTED_SHAPES - set(seen))
    status = "FAIL" if audit.errors else ("INCOMPLETE" if missing else "PASS")
    print(json.dumps(dict(
        status=status, completed_case_records=len(loaded), expected_cases=18,
        missing_shapes=missing, expected_candidate_attempts=18*66,
        measured_candidates=measured_candidates, skipped_candidates=skipped_candidates,
        final_backend_records=sum(len(case["results"]) for case in loaded),
        raw_replay_samples_audited=audit.raw_replay_samples,
        max_recorded_abs_error=audit.max_recorded_abs_error,
        fresh_input_protocol="Hashed script and stored checks audited; inputs are not retained, so no independent GEMM recomputation.",
        notes=["No GPU calls or measurement-module imports.",
               "Recorded timings were recomputed from saved per-replay microseconds; CUDA event timestamps are not retained.",
               "Full-output finite and zero/restore assertions are verified in the script, not separate serialized flags.",
               "Overall optimized winner and one family winner reuse the same graph and are timed separately.",
               "Family winners can select different tiles; same-tile ablations require matching config_id modulo 8."],
        warnings=audit.warnings, errors=audit.errors), indent=2))
    return 1 if audit.errors else (2 if missing else 0)


if __name__ == "__main__":
    raise SystemExit(main())
