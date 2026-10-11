import argparse
import ctypes
import json
import statistics
import sys

import torch

sys.path[:0] = ["/workspace/grain-gemm/src", "/workspace/grain-gemm/tests"]
from test_int8_gemm import inputs, reference

parser = argparse.ArgumentParser()
parser.add_argument("--only-check", action="store_true")
args = parser.parse_args()
lib = ctypes.CDLL("/tmp/thor-ws2-race-syncsf/probe.so")
lib.probe.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int]
lib.probe.restype = ctypes.c_int


def run(tensors, output, stages):
    a, b, sa, sb = tensors
    code = lib.probe(
        *(ctypes.c_void_p(t.data_ptr()) for t in (*tensors, output)),
        a.shape[0], b.shape[1], a.shape[1],
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), stages,
    )
    if code:
        raise RuntimeError(code)


for shape in [(256, 128, 256), (256, 128, 768), (256, 384, 512),
              (256, 384, 768), (256, 384, 4352), (512, 640, 768)]:
    m, n, k = shape
    cpu = list(inputs(*shape, 256, layout="column_major_b"))
    groups = torch.arange(k // 256)
    cpu[2].copy_(((torch.arange(m)[:, None] * 5 + groups * 7) % 31 - 15).float() * 0.03125)
    cpu[3].copy_(((torch.arange(n)[None, :] * 11 + groups[:, None] * 13) % 37 - 18).float() * 0.0625)
    expected = reference(*cpu, 256).bfloat16()
    tensors = tuple(t.to("cuda") for t in cpu)
    guard = 32
    storage = torch.full((m * n + 2 * guard,), 13, dtype=torch.bfloat16, device="cuda")
    output = storage[guard:-guard].view(m, n)
    for stages in (2, 3):
        print("launch", shape, stages, flush=True)
        run(tensors, output, stages)
        torch.testing.assert_close(output.cpu(), expected)
        assert torch.all(storage[:guard] == 13) and torch.all(storage[-guard:] == 13)
        print("PASS", shape, stages, "output guards intact", flush=True)

    if shape == (256, 384, 768):
        for stages in (2, 3):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run(tensors, output, stages)
            tensors[0].neg_()
            tensors[3].mul_(-2)
            graph.replay()
            expected_updated = reference(*tensors, 256).bfloat16()
            torch.testing.assert_close(output.cpu(), expected_updated)
            assert torch.all(storage[:guard] == 13) and torch.all(storage[-guard:] == 13)
            print("Graph update PASS", stages, flush=True)

if args.only_check:
    raise SystemExit(0)

from grain_gemm.kernels import cutlass
from triton.testing import do_bench_cudagraph

warm = inputs(4096, 4096, 4096, 256, device="cuda", layout="column_major_b")
warm_out = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
do_bench_cudagraph(lambda: cutlass.launch(*warm, warm_out, 32), rep=1000)
results = []
for shape in [(512, 3072, 1536), (1024, 1024, 1024), (2048, 2048, 2048),
              (2048, 151936, 1536)]:
    tensors = inputs(*shape, 256, device="cuda", layout="column_major_b")
    output = torch.empty(shape[:2], device="cuda", dtype=torch.bfloat16)
    functions = {"ws2_p2": lambda: run(tensors, output, 2),
                 "ws2_p3": lambda: run(tensors, output, 3),
                 "cuts32": lambda: cutlass.launch(*tensors, output, 32)}
    samples = {name: [] for name in functions}
    for iteration in range(3):
        names = list(functions)
        names = names[iteration:] + names[:iteration]
        for name in names:
            functions[name]()
            torch.cuda.synchronize()
            samples[name].append(do_bench_cudagraph(functions[name], rep=100) * 1000)
    result = {"shape": shape,
              "median_us": {name: statistics.median(values) for name, values in samples.items()},
              "samples_us": samples}
    print(result, flush=True)
    results.append(result)
with open("/tmp/thor-ws2-tail-probe/results.json", "w") as handle:
    json.dump(results, handle, indent=2)
    handle.write("\n")
