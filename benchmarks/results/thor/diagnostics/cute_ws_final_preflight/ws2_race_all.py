import ctypes
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--lib", default="/tmp/thor-ws2-race-probe/probe_full32_zfill.so")
args = parser.parse_args()
print("library", args.lib, flush=True)
lib = ctypes.CDLL(args.lib)
lib.probe.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int]
lib.probe.restype = ctypes.c_int
shapes = [(256, 256, 256), (256, 512, 512), (256, 512, 4352),
          (1024, 1024, 768), (768, 1152, 768), (2816, 384, 768),
          (256, 128, 768), (256, 384, 4352)]
for m, n, k in shapes:
    a = torch.full((m, k), -128, device="cuda", dtype=torch.int8)
    b = torch.full((n, k), 127, device="cuda", dtype=torch.int8).t()
    sa = torch.full((m, k // 256), 0.125, device="cuda")
    sb = torch.full((k // 256, n), 0.25, device="cuda")
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for p in (2, 3):
        print("launch", m, n, k, p, flush=True)
        code = lib.probe(*(ctypes.c_void_p(x.data_ptr()) for x in (a, b, sa, sb, c)),
                         m, n, k, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), p)
        if code:
            raise RuntimeError(code)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, torch.full_like(c, -128 * 127 * k * 0.125 * 0.25),
                                   rtol=0, atol=0)
        print("PASS", flush=True)
