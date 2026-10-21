#!/usr/bin/env python3
"""Single G256 INT8 GEMM: public auto and native CUTLASS candidates on Thor.

Uses the read-only clock and thermal guards from bench_batched_gemm.py.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bench_batched_gemm import Guard, sampled_reference, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[1024, 2048, 4096, 8192])
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=1536)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.n, args.k) <= 0 or args.k % 256 or args.n % 256 or any(m % 256 for m in args.rows):
        parser.error('Positive M/N/K multiples of 256 required')
    import torch
    from grain_gemm import int8_gemm, get_kernel_config
    from grain_gemm.kernels import cuda, cutlass, cutlass_moe
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability() != (11, 0) or not cuda.is_available() or not cutlass.is_available() or not cutlass_moe.is_available():
        raise RuntimeError('Matching Thor native builds required')
    package = Path(__file__).resolve().parents[1] / 'src/grain_gemm'
    tracked = {p for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h','.hpp','.json','.so')}
    tracked.update((Path(__file__).resolve(), Path(__file__).with_name('bench_batched_gemm.py')))
    def fingerprints():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)}
    before = fingerprints()
    guard = Guard(torch)
    report = dict(complete=False, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  arguments=vars(args) | {'output':str(args.output)}, source_before=before,
                  hardware=dict(gpu=torch.cuda.get_device_name(),gpu_hz=1197000000,emc_hz=2750000000),
                  software=dict(torch=torch.__version__,cuda=torch.version.cuda), cases=[])
    save(report,args.output)
    try:
        with torch.inference_mode():
            for m in args.rows:
                guard.cool()
                n,k=args.n,args.k
                torch.manual_seed(2026+m+n+k)
                a=torch.randint(-128,128,(1,m,k),device='cuda',dtype=torch.int8)
                w=torch.randint(-128,128,(1,n,k),device='cuda',dtype=torch.int8)
                sa=torch.rand((1,m,k//256),device='cuda')*.003+.001
                sb=torch.rand((1,k//256,n),device='cuda')*.003+.001
                sa[:,0,0]=0; sb[:,0,::17]*=-1
                out=torch.empty((1,m,n),device='cuda',dtype=torch.bfloat16)
                indices,expected=sampled_reference(a,w,sa,sb,torch)
                ei,ri,ci=indices
                graphs,checks,functions={},{},{}
                auto_config=get_kernel_config(a[0],w[0].T,scale_a=sa[0],scale_b=sb[0])
                def auto():
                    return int8_gemm(a[0],w[0].T,sa[0],sb[0],output_dtype=torch.bfloat16)
                functions['auto']=auto
                for config,cfg in enumerate(cuda.configs_for_device(a.device)):
                    if m % cfg['block_m'] or n % cfg.get('native_n_multiple',cfg['block_n']):
                        continue
                    functions[f'cute_{config}']=lambda config=config:cuda.launch(a[0],w[0].T,sa[0],sb[0],out[0],config)
                for config in range(len(cutlass.configs_for_device(a.device))):
                    functions[f'cutlass_{config}']=lambda config=config:cutlass.launch(a[0],w[0].T,sa[0],sb[0],out[0],config)
                for config in cutlass_moe.CONFIGS:
                    functions[f'expert_{config}']=lambda config=config:cutlass_moe.launch(a,w,sa,sb,out,config)
                for name,fn in functions.items():
                    result=guard.call(fn)
                    actual=result.reshape(1,m,n)[ei[:,None,None],ri[None,:,None],ci[None,None,:]].cpu()
                    torch.testing.assert_close(actual,expected,rtol=1/128,atol=1e-3)
                    checks[name]=dict(sampled_elements=expected.numel(),bf16_bits_equal=torch.equal(actual.view(torch.int16),expected.view(torch.int16)),max_abs_error=(actual.float()-expected.float()).abs().max().item())
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured=fn()
                    guard.call(graph.replay)
                    actual=captured.reshape(1,m,n)[ei[:,None,None],ri[None,:,None],ci[None,None,:]].cpu()
                    torch.testing.assert_close(actual,expected,rtol=1/128,atol=1e-3)
                    graphs[name]=graph
                screen=guard.measure(graphs,rounds=3,samples=1)
                winner=min(screen,key=lambda v:screen[v]['median_ms'])
                warm_screen=guard.measure(graphs,rounds=3,samples=1,cold=False)
                warm_winner=min(warm_screen,key=lambda v:warm_screen[v]['median_ms'])
                cute_cold=min((name for name in screen if name.startswith('cute_')),key=lambda name:screen[name]['median_ms'])
                cute_warm=min((name for name in warm_screen if name.startswith('cute_')),key=lambda name:warm_screen[name]['median_ms'])
                selected={name:graphs[name] for name in dict.fromkeys(('auto',winner,warm_winner,cute_cold,cute_warm))}
                results={}
                for mode,cold in (('cold',True),('warm',False)):
                    for g in selected.values():guard.call(g.replay)
                    values=guard.measure(selected,rounds=3,samples=3,cold=cold)
                    for value in values.values():value['tops']=2*m*n*k/(value['median_ms']*1e9)
                    results[mode]=values
                result=guard.call(functions[winner])
                actual=result.reshape(1,m,n)[ei[:,None,None],ri[None,:,None],ci[None,None,:]].cpu()
                torch.testing.assert_close(actual,expected,rtol=1/128,atol=1e-3)
                assert bool(guard.call(lambda:torch.isfinite(result).all()).item())
                case=dict(m=m,n=n,k=k,winner=winner,warm_winner=warm_winner,cute_cold=cute_cold,cute_warm=cute_warm,auto_config=auto_config,checks=checks,
                          screening=screen,warm_screening=warm_screen,results=results,output_all_finite=True)
                report['cases'].append(case);save(report,args.output)
                print(m,winner,{mode:{name:dict(ms=round(v['median_ms'],6),tops=round(v['tops'],2)) for name,v in values.items()} for mode,values in results.items()},flush=True)
                del graphs,selected,functions,a,w,sa,sb,out,result,captured
        report['complete']=True
    finally:
        report['source_after']=fingerprints();report['provenance_unchanged']=report['source_after']==before
        report['states']=guard.states
        report['maximum_temperature_c']=max(s['temperature_c'] for s in guard.states)
        save(report,args.output)
        if not report['provenance_unchanged']:raise RuntimeError('Measured sources or native builds changed')


if __name__=='__main__':
    main()
