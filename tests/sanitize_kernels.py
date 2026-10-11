"""Compute Sanitizer smoke checks for native pipeline boundaries and Triton tails.

Requires the optional native build on GB10. Run from the repository root:
compute-sanitizer --tool memcheck --error-exitcode 1 python tests/sanitize_kernels.py
compute-sanitizer --tool racecheck --error-exitcode 1 python tests/sanitize_kernels.py
"""

import torch
from grain_gemm import int8_gemm
from grain_gemm.kernels import cuda


def main():
    for k in (256, 4352):
     for config in range(len(cuda.configs_for_device())):
      m,n=256,256
      a=torch.full((m,k),-128,device='cuda',dtype=torch.int8)
      b=torch.full((n,k),127,device='cuda',dtype=torch.int8).T
      sa=torch.full((m,k//256),.125,device='cuda')
      sb=torch.full((k//256,n),.25,device='cuda')
      c=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
      cuda.launch(a,b,sa,sb,c,config)
      torch.cuda.synchronize()
      torch.testing.assert_close(c, torch.full_like(c,-128*127*k*.125*.25),rtol=0,atol=0)
    if torch.cuda.get_device_capability() == (11, 0):
        # Sixteen two-SM tiles exceed Thor's ten resident clusters. Odd group
        # counts exercise accumulator/ring phases across persistent tile reuse.
        m, n, k = 1024, 1024, 768
        a = torch.full((m, k), -128, device='cuda', dtype=torch.int8)
        b = torch.full((n, k), 127, device='cuda', dtype=torch.int8).T
        sa = torch.full((m, k // 256), .125, device='cuda')
        sb = torch.full((k // 256, n), .25, device='cuda')
        c = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        for config, descriptor in enumerate(cuda.SM110_CONFIGS):
            if descriptor.get('mma') != 'tcgen05_ws':
                continue
            cuda.launch(a, b, sa, sb, c, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                c, torch.full_like(c, -128 * 127 * k * .125 * .25),
                rtol=0, atol=0,
            )
        # N128 half-tiles also reuse a persistent cluster; the inactive weight
        # scale half and output half must never access global memory.
        m, n = 768, 1152
        a = torch.full((m, k), -128, device='cuda', dtype=torch.int8)
        b = torch.full((n, k), 127, device='cuda', dtype=torch.int8).T
        sa = torch.full((m, k // 256), .125, device='cuda')
        sb = torch.full((k // 256, n), .25, device='cuda')
        c = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        for config, descriptor in enumerate(cuda.SM110_CONFIGS):
            if descriptor.get('cluster_m') != 2:
                continue
            cuda.launch(a, b, sa, sb, c, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                c, torch.full_like(c, -128 * 127 * k * .125 * .25),
                rtol=0, atol=0,
            )
    # Exercise masked generic addressing and positive-stride views as well.
    a=torch.ones((19,269),device='cuda',dtype=torch.int8)
    b=torch.ones((269,37),device='cuda',dtype=torch.int8)
    sa=torch.ones((19,2),device='cuda');sb=torch.ones((2,37),device='cuda')
    c=int8_gemm(a,b,sa,sb,group_size=256,output_dtype=torch.float32,backend='triton')
    torch.cuda.synchronize()
    torch.testing.assert_close(c,torch.full_like(c,269))
    print('Native pipeline boundaries and Triton tails passed')


if __name__ == "__main__":
    main()
