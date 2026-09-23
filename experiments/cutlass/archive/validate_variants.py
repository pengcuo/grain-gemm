"""Full-output extreme-value and pipeline-wrap checks for isolated variants."""

import sys

import torch

from paired_benchmark import Native, ROOT

sys.path.insert(0, '/home/pengcuo/spark/grain-gemm/tests')
from sanitize_cutlass import case


variants = sys.argv[1:] or ['baseline', 'no_direct_tail_barrier', 'bf16_uint4_i32_address']
libraries = [Native(ROOT / name / 'libgrain_cutlass.so', 'grain_cutlass_g256')
             for name in variants]
checked = 0
rejected = 0
for m, n, k in ((384, 256, 256), (256, 384, 768),
                (384, 256, 1280), (384, 256, 4352)):
    a, b, sa, sb, expected = case(m, n, k)
    out = torch.empty_like(expected)
    for library in libraries:
        for config in (11, 15, 23):
            if library.path.parent.name == 'ungrouped_specialization' and k > 4096:
                try:
                    library.launch(a, b, sa, sb, out, config, torch)
                except RuntimeError:
                    rejected += 1
                    continue
                raise AssertionError('Ungrouped experimental kernel accepted unsupported K')
            library.launch(a, b, sa, sb, out, config, torch)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            checked += 1
print(f'{checked} launches passed full-output reference checks; {rejected} unsupported launches rejected')
