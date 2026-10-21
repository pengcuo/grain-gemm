#!/usr/bin/env python3
"""Paired CuTe INT8 / default torch.mm BF16 GEMM with read-only EMC guards."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from bench_batched_gemm import Guard, sampled_reference, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emc-dynamic',action='store_true')
    parser.add_argument('--emc-hz',type=int,default=2750000000)
    parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--k',type=int,default=1536)
    parser.add_argument('--rows',type=int,nargs='+',default=[1024,2048,4096,8192])
    parser.add_argument('--cute-config',type=int,default=27)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.n,args.k,*args.rows)<=0 or any(v%256 for v in (args.n,args.k,*args.rows)):
        parser.error('M/N/K must be positive multiples of 256')
    import torch
    from grain_gemm.kernels import cuda
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability()!=(11,0) or not cuda.is_available():
        raise RuntimeError('Thor with matching CuTe native build required')
    package=Path(__file__).resolve().parents[1]/'src/grain_gemm'
    tracked={p for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h','.hpp','.json','.so')}
    tracked.update((Path(__file__).resolve(),Path(__file__).with_name('bench_batched_gemm.py')))
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)}
    guard=Guard(torch,emc_dynamic=args.emc_dynamic,emc_hz=args.emc_hz)
    report=dict(complete=False,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                emc_policy='dynamic' if args.emc_dynamic else f'locked_{args.emc_hz}',gpu_hz=1197000000,
                software=dict(torch=torch.__version__,cuda=torch.version.cuda),source_before=hashes(),
                blas_preference=str(torch.backends.cuda.preferred_blas_library()),
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                cases=[])
    save(report,args.output)
    try:
        with torch.inference_mode():
            for m in args.rows:
                guard.cool();n,k=args.n,args.k
                torch.manual_seed(2026+m+n+k)
                x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
                def quantize(t):
                    groups=t.float().reshape(t.shape[0],-1,256)
                    scale=groups.abs().amax(-1).clamp_min(1e-12)/127
                    return (groups/scale[:,:,None]).round().clamp(-127,127).to(torch.int8).reshape_as(t),scale
                a,sa=quantize(x);wi,sw=quantize(w);sb=sw.T.contiguous()
                oi=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);ob=torch.empty_like(oi)
                functions=dict(cute_int8=lambda:cuda.launch(a,wi.T,sa,sb,oi,args.cute_config),
                               torch_bf16=lambda:torch.mm(x,w.T,out=ob))
                indices,expected=sampled_reference(a[None],wi[None],sa[None],sb[None],torch)
                _,rows,cols=indices
                reference=x[rows].cpu().double()@w[cols].cpu().double().T
                def check():
                    actual=oi[rows[:,None],cols[None,:]].cpu()
                    assert torch.equal(actual.view(torch.int16),expected[0].view(torch.int16))
                    actual_bf=ob[rows[:,None],cols[None,:]].cpu().double()
                    torch.testing.assert_close(actual_bf,reference,rtol=1/128,atol=.05)
                    for out in (oi,ob):assert bool(guard.call(lambda:torch.isfinite(out).all()).item())
                graphs={}
                for name,fn in functions.items():
                    guard.call(fn);graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):fn()
                    guard.call(graph.replay);graphs[name]=graph
                check();results={}
                for mode,cold in (('cold',True),('warm',False)):
                    for graph in graphs.values():guard.call(graph.replay)
                    if cold:
                        values=guard.measure(graphs,rounds=5,samples=5,cold=True)
                    else:
                        values={}
                        for name,graph in graphs.items():
                            guard.call(graph.replay)
                            values.update(guard.measure({name:graph},rounds=5,samples=5,cold=False))
                    for value in values.values():value['throughput']=2*m*n*k/(value['median_ms']*1e9)
                    results[mode]=values
                check()
                report['cases'].append(dict(m=m,n=n,k=k,cute_config=args.cute_config,results=results,
                    int8_sampled_bits_equal=True,bf16_sampled_reference_pass=True,outputs_all_finite=True))
                save(report,args.output)
                print(m,{mode:{name:dict(us=round(v['median_ms']*1000,3),throughput=round(v['throughput'],2)) for name,v in values.items()} for mode,values in results.items()},flush=True)
        report['complete']=True
    finally:
        report['source_after']=hashes();report['provenance_unchanged']=report['source_after']==report['source_before']
        report['states']=guard.states
        report['maximum_temperature_c']=max(s['temperature_c'] for s in guard.states)
        save(report,args.output)
        if not report['provenance_unchanged']:raise RuntimeError('Source/native files changed')


if __name__=='__main__':main()
