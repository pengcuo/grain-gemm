#!/usr/bin/env python3
"""Default torch.mm BF16 rectangular GEMM with Thor clock/thermal guards."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from bench_batched_gemm import Guard, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[1024,2048,4096,8192])
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=1536)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if min(*args.rows,args.n,args.k)<=0:
        parser.error('M/N/K must be positive')
    import torch
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability() != (11,0):
        raise RuntimeError('Thor SM110 required')
    tracked=(Path(__file__).resolve(),Path(__file__).with_name('bench_batched_gemm.py'))
    def hashes():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    guard=Guard(torch)
    report=dict(complete=False,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                arguments=vars(args)|{'output':str(args.output)},source_before=hashes(),
                hardware=dict(gpu=torch.cuda.get_device_name(),gpu_hz=1197000000,emc_hz=2750000000),
                software=dict(torch=torch.__version__,cuda=torch.version.cuda),
                blas_preference=str(torch.backends.cuda.preferred_blas_library()),
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                float32_matmul_precision=torch.get_float32_matmul_precision(),cases=[])
    save(report,args.output)
    try:
        with torch.inference_mode():
            for m in args.rows:
                guard.cool();n,k=args.n,args.k
                torch.manual_seed(2026+m+n+k)
                a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                weight=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
                out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                def run():
                    return torch.mm(a,weight.T,out=out)
                guard.call(run)
                rows=torch.tensor([0,m//2,m-1],device='cuda')
                cols=torch.linspace(0,n-1,9,device='cuda').long().unique()
                expected=a[rows].cpu().double() @ weight[cols].cpu().double().T
                def check():
                    actual=out[rows[:,None],cols[None,:]].cpu().double()
                    torch.testing.assert_close(actual,expected,rtol=1/128,atol=.05)
                    assert bool(guard.call(lambda:torch.isfinite(out).all()).item())
                    return (actual-expected).abs().max().item()
                error=check()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):run()
                guard.call(graph.replay)
                results={}
                for mode,cold in (('cold',True),('warm',False)):
                    guard.call(graph.replay)
                    value=guard.measure({'torch_mm':graph},rounds=5,samples=5,cold=cold)['torch_mm']
                    value['tflops']=2*m*n*k/(value['median_ms']*1e9)
                    results[mode]=value
                guard.call(graph.replay);error=max(error,check())
                case=dict(m=m,n=n,k=k,results=results,sampled_elements=expected.numel(),
                          sampled_max_abs_error_vs_cpu_fp64=error,output_all_finite=True)
                report['cases'].append(case);save(report,args.output)
                print(m,{mode:dict(us=round(v['median_ms']*1000,3),tflops=round(v['tflops'],2)) for mode,v in results.items()},flush=True)
        report['complete']=True
    finally:
        report['source_after']=hashes();report['provenance_unchanged']=report['source_after']==report['source_before']
        report['states']=guard.states
        report['maximum_temperature_c']=max(s['temperature_c'] for s in guard.states)
        save(report,args.output)
        if not report['provenance_unchanged']:raise RuntimeError('Benchmark sources changed')


if __name__=='__main__':
    main()
