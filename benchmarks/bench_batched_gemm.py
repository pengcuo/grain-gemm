#!/usr/bin/env python3
"""Equal-length expert batched G256 INT8 GEMM on Thor; no MoE routing or activation.

PYTHONPATH=src python benchmarks/bench_batched_gemm.py --output /tmp/batched_gemm.json
Clocks are verified read-only. Both cold-cache and repeated-input warm results
are recorded; raw measurements should be written outside the repository.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GPU_HZ, EMC_HZ = 1197000000, 2750000000


class Guard:
    def __init__(self, torch, *, emc_dynamic=False, emc_hz=EMC_HZ):
        self.torch = torch
        self.emc_dynamic = emc_dynamic
        self.emc_hz = emc_hz
        self.emc_frequencies = set(map(int, (Path("/sys/class/devfreq/bwmgr/available_frequencies")).read_text().split()))
        self.sensors = [p / "temp" for p in Path("/sys/class/thermal").glob("thermal_zone*")
                        if "gpu" in (p / "type").read_text().lower()]
        if not self.sensors:
            raise RuntimeError("GPU temperature sensor is required")
        self.states = []
        self.flush = torch.empty(256_000_000, device="cuda", dtype=torch.uint8)

    def state(self):
        clocks = {f"{dev}_{key}": int((Path("/sys/class/devfreq") / dev / key).read_text())
                  for dev in ("gpu-gpc-0", "bwmgr") for key in ("min_freq", "max_freq", "cur_freq")}
        expected_min = min(self.emc_frequencies) if self.emc_dynamic else self.emc_hz
        expected_max = max(self.emc_frequencies) if self.emc_dynamic else self.emc_hz
        if (any(value != GPU_HZ for key, value in clocks.items() if key.startswith("gpu"))
                or clocks["bwmgr_min_freq"] != expected_min
                or clocks["bwmgr_max_freq"] != expected_max
                or clocks["bwmgr_cur_freq"] not in self.emc_frequencies
                or not expected_min <= clocks["bwmgr_cur_freq"] <= expected_max):
            raise RuntimeError(f"Require GPU 1197 MHz and EMC range {expected_min}..{expected_max}: {clocks}")
        temp = max(int(p.read_text()) / 1000 for p in self.sensors)
        if temp >= 70:
            raise RuntimeError(f"GPU temperature guard: {temp} C")
        value = dict(clocks=clocks, temperature_c=temp)
        self.states.append(value)
        return value

    def cool(self):
        if self.state()["temperature_c"] < 62:
            return
        self.torch.cuda.synchronize()
        started = time.monotonic()
        while self.state()["temperature_c"] > 54:
            if time.monotonic() - started > 120:
                raise RuntimeError("Cooling timeout")
            time.sleep(.25)

    def call(self, function):
        self.cool()
        result = function()
        self.torch.cuda.synchronize()
        self.state()
        return result

    def measure(self, graphs, *, rounds, samples, cold=True):
        torch = self.torch
        values = {name: [] for name in graphs}
        names = list(graphs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for rd in range(rounds):
            row = {name: [] for name in names}
            for sample in range(samples):
                shift = (rd + sample) % len(names)
                order = names[shift:] + names[:shift]
                if sample % 2:
                    order = order[::-1]
                for name in order:
                    self.cool()
                    if cold:
                        self.flush.zero_()
                    torch.cuda.synchronize()
                    before = self.state()
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    row[name].append(ms)
                    values[name].append(dict(round=rd, sample=sample, ms=ms,
                                             before=before, after=self.state()))
            for name in names:
                values[name][-1]["round_median_ms"] = statistics.median(row[name])
        return {name: dict(samples=items,
                           round_medians_ms=[v["round_median_ms"] for v in items if "round_median_ms" in v],
                           median_ms=statistics.median(v["round_median_ms"] for v in items if "round_median_ms" in v))
                for name, items in values.items()}


def batched_gemm(a, b, scale_a, scale_b, out, *, config_id=45, swizzle=1, raster=0):
    """A[E,M,K] @ B[E,K,N], independent G256 scales, BF16 output[E,M,N].

    B is the transposed view of contiguous Linear weights [E,N,K]. Scales
    are contiguous SA[E,M,K/256] and SB[E,K/256,N]. One native batched launch
    covers every expert. All rows are valid; counts=None and no routing exists.
    """
    from grain_gemm.kernels import cutlass_moe
    return cutlass_moe.launch(a, b.transpose(-2, -1), scale_a, scale_b, out,
                             config_id, swizzle=swizzle, raster=raster)


def sampled_reference(a, weight, sa, sb, torch):
    """Independent CPU INT32 group dots + FP32 scaled FMA rounding."""
    e, m, k = a.shape
    n = weight.shape[1]
    experts = torch.tensor(sorted({0, e // 2, e - 1}), device="cuda")
    rows = torch.tensor(sorted({0, m // 2, m - 1}), device="cuda")
    cols = torch.linspace(0, n - 1, 9, device="cuda").long().unique()
    aa = a[experts[:, None, None], rows[None, :, None], torch.arange(k, device="cuda")[None, None, :]].cpu()
    bb = weight[experts[:, None, None], cols[None, :, None], torch.arange(k, device="cuda")[None, None, :]].cpu()
    saa = sa[experts[:, None, None], rows[None, :, None], torch.arange(k // 256, device="cuda")[None, None, :]].cpu()
    sbb = sb[experts[:, None, None], torch.arange(k // 256, device="cuda")[None, :, None], cols[None, None, :]].cpu()
    result = torch.zeros((len(experts), len(rows), len(cols)), dtype=torch.float32)
    for group, start in enumerate(range(0, k, 256)):
        partial = aa[..., start:start + 256].int() @ bb[..., start:start + 256].int().transpose(-2, -1)
        scale = saa[:, :, group, None] * sbb[:, group, None, :]
        result = (result.double() + partial.double() * scale.double()).float()
    return (experts, rows, cols), result.bfloat16()


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.experts, args.hidden, args.intermediate, *args.rows) <= 0
            or args.hidden % 256 or args.intermediate % 256 or any(m % 256 for m in args.rows)):
        parser.error("Positive sizes required; H/I/M must be multiples of 256")
    import torch
    from grain_gemm.kernels import cutlass_moe
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability() != (11, 0) or not cutlass_moe.is_available():
        raise RuntimeError("Require Thor and tools/build_cutlass_moe.py --cutlass-dir PATH")
    tracked = {ROOT / "src/grain_gemm/kernels/cutlass_moe.py", Path(__file__).resolve(), cutlass_moe._LIBRARY,
               cutlass_moe._LIBRARY.with_name("moe_cutlass_build.json")}
    tracked.update(ROOT / "src/grain_gemm/kernels/csrc" / p for p in cutlass_moe.source_hashes())
    def fingerprints():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)}
    initial = fingerprints()
    guard = Guard(torch)
    report = dict(complete=False, cases=[], arguments=vars(args) | {"output": str(args.output)},
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope="One equal-length E-expert G256 INT8 batched GEMM, BF16 output, no routing/packing/quantization/activation/combine. Input preparation excluded.",
                  timing="One GEMM per CUDA graph replay; 256MB flush outside cold-cache events. Warm repeated-input confirmation has no flush. No long sustained warmup.",
                  throughput="2*E*M*N*K / time, INT8 TOPS; no sparsity multiplier", source_before=initial,
                  hardware=dict(gpu=torch.cuda.get_device_name(), gpu_hz=GPU_HZ, emc_hz=EMC_HZ),
                  software=dict(torch=torch.__version__, cuda=torch.version.cuda), configs=cutlass_moe.CONFIGS)
    save(report, args.output)
    try:
        with torch.inference_mode():
            for m in args.rows:
                for name, n, k in (("gate_up", 2 * args.intermediate, args.hidden),
                                   ("down", args.hidden, args.intermediate)):
                    guard.cool()
                    torch.manual_seed(2026 + m + n + k)
                    e = args.experts
                    a = torch.randint(-128, 128, (e,m,k), device="cuda", dtype=torch.int8)
                    w = torch.randint(-128, 128, (e,n,k), device="cuda", dtype=torch.int8)
                    sa = torch.rand((e,m,k//256), device="cuda") * .003 + .001
                    sb = torch.rand((e,k//256,n), device="cuda") * .003 + .001
                    sa[:,0,0] = 0
                    sb[:,0,::17] *= -1
                    out = torch.empty((e,m,n), device="cuda", dtype=torch.bfloat16)
                    indices, expected = sampled_reference(a,w,sa,sb,torch)
                    guard.state()
                    experts, rows, cols = indices
                    graphs, candidates = {}, {}
                    def build(config, swizzle=1, raster=0):
                        key = f"c{config}_s{swizzle}_r{raster}"
                        if key in graphs:
                            return key
                        guard.call(lambda:batched_gemm(a,w.transpose(-2,-1),sa,sb,out,
                                                       config_id=config,swizzle=swizzle,raster=raster))
                        def check():
                            actual = out[experts[:,None,None],rows[None,:,None],cols[None,None,:]].cpu()
                            if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                                raise AssertionError(f"CPU reference mismatch: {m}/{name}/{key}")
                        check()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            cutlass_moe._launch_prevalidated(a,w,sa,sb,out,config,swizzle=swizzle,raster=raster)
                        guard.call(graph.replay)
                        check()
                        graphs[key] = graph
                        candidates[key] = dict(config=config, swizzle=swizzle, raster=raster)
                        return key
                    for config in cutlass_moe.CONFIGS:
                        build(config)
                    screen = guard.measure(graphs, rounds=3, samples=1)
                    leaders = sorted(screen, key=lambda key:screen[key]["median_ms"])[:2]
                    for leader in leaders:
                        config = candidates[leader]["config"]
                        for swizzle in (1,4,8):
                            for raster in (0,1,2):
                                build(config,swizzle,raster)
                    tuned = guard.measure(graphs, rounds=3, samples=1)
                    winner = min(tuned, key=lambda key:tuned[key]["median_ms"])
                    results = {}
                    for mode, cold in (("cold", True),("warm", False)):
                        guard.call(graphs[winner].replay)
                        result = guard.measure({winner:graphs[winner]}, rounds=3, samples=3, cold=cold)[winner]
                        result["tops"] = 2*e*m*n*k / (result["median_ms"]*1e9)
                        results[mode] = result
                    guard.call(graphs[winner].replay)
                    actual = out[experts[:,None,None],rows[None,:,None],cols[None,None,:]].cpu()
                    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
                    finite = bool(guard.call(lambda:torch.isfinite(out).all()).item())
                    assert finite
                    case = dict(projection=name, experts=e, rows_per_expert=m, n=n, k=k,
                                input_tokens_if_top4=m*e//4, operations=2*e*m*n*k,
                                config=candidates[winner], winner=winner, sampled_elements=expected.numel(),
                                cpu_reference_bits_equal=True, output_all_finite=finite,
                                initial_screen=screen, tuned_screen=tuned, results=results)
                    report["cases"].append(case)
                    print(m,name,winner,{mode:dict(ms=round(value["median_ms"],6),tops=round(value["tops"],2))
                                         for mode,value in results.items()},flush=True)
                    save(report,args.output)
                    del graphs, a, w, sa, sb, out
        report["complete"] = True
    finally:
        report["source_after"] = fingerprints()
        report["provenance_unchanged"] = report["source_after"] == initial
        report["states"] = guard.states
        report["maximum_temperature_c"] = max(v["temperature_c"] for v in guard.states)
        save(report,args.output)
        if not report["provenance_unchanged"]:
            raise RuntimeError("Benchmark source or native build changed during measurement")


if __name__ == "__main__":
    main()
