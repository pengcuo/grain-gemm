import ctypes
import json
import statistics
import sys

import torch

sys.path[:0] = ["/workspace/grain-gemm/src", "/workspace/grain-gemm/tests"]
from test_int8_gemm import inputs, reference
from grain_gemm.kernels import cutlass
from triton.testing import do_bench_cudagraph

lib = ctypes.CDLL("/tmp/thor-ws2-race-syncsf/probe.so")
lib.probe.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int]
lib.probe.restype = ctypes.c_int


def run(tensors, output, stages):
    a, b, sa, sb = tensors
    code = lib.probe(*(ctypes.c_void_p(t.data_ptr()) for t in (*tensors, output)),
                     a.shape[0], b.shape[1], a.shape[1],
                     ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), stages)
    if code:
        raise RuntimeError(code)


warm = inputs(4096, 4096, 4096, 256, device="cuda", layout="column_major_b")
warm_out = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
do_bench_cudagraph(lambda: cutlass.launch(*warm, warm_out, 32), rep=1000)
results = []
for shape in [(1024, 1024, 1024), (4096, 4096, 4096), (2048, 151936, 1536)]:
    tensors = inputs(*shape, 256, device="cuda", layout="column_major_b")
    output = torch.empty(shape[:2], device="cuda", dtype=torch.bfloat16)
    if shape[1] % 256:
        rows = [0, 127, 128, 255, 256, shape[0] - 1]
        cols = [0, 127, 128, 255, shape[1] - 129, shape[1] - 128, shape[1] - 1]
        a, b, sa, sb = tensors
        expected = reference(a[rows], b[:, cols], sa[rows], sb[:, cols], 256).bfloat16()
        for stages in (2, 3):
            run(tensors, output, stages)
            torch.testing.assert_close(output[rows][:, cols].cpu(), expected)
            print("wide sampled CPU reference PASS", shape, stages, flush=True)
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
with open("/tmp/thor-ws2-race-syncsf/results.json", "w") as handle:
    json.dump(results, handle, indent=2)
    handle.write("\n")
