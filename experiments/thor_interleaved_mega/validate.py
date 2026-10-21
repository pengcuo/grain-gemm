#!/usr/bin/env python3
"""Targeted root-owned Thor Mega GPU checks; import/help executes no GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_current(function,guard,implementation):
    native=implementation._native_gemm
    data={}
    def wrapped(*args,**kwargs):
        result=native(*args,**kwargs)
        data.update(qh=args[0].clone(),sh=args[2].clone(),expert_y=args[4].clone(),counts=kwargs['counts'].clone())
        return result
    with patch.object(implementation,'_native_gemm',wrapped):
        data['output']=guard.call(function)
    return data


def exact_checks(actual,reference,expected_counts,torch):
    checks=dict(output_bits_equal=bool(torch.equal(actual['output'].view(torch.int16),reference['output'].view(torch.int16))),
        counts_equal=bool(torch.equal(actual['counts'],expected_counts)),all_finite=bool(torch.isfinite(actual['output']).all()))
    if 'qh' in reference:
        checks['qh_bits_equal']=bool(torch.equal(actual['qh'],reference['qh']))
        checks['sh_bits_equal']=bool(torch.equal(actual['sh'].view(torch.int32),reference['sh'].view(torch.int32)))
        valid=torch.arange(actual['capacity'],device=actual['counts'].device)[None,:]<expected_counts[:,None]
        checks['expert_y_useful_bits_equal']=bool(torch.equal(actual['expert_y'][valid].view(torch.int16),reference['expert_y'][valid].view(torch.int16)))
        checks['padding_q_zero']=bool((actual['qh'][~valid]==0).all())
        zero_sf=(torch.tensor(1.e-12,dtype=torch.float32)/torch.tensor(127.,dtype=torch.float32)).to(actual['sh'].device)
        checks['padding_scale_bits_equal']=bool((actual['sh'][~valid].view(torch.int32)==zero_sf.view(torch.int32)).all())
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'build/validation.json')
    args=parser.parse_args()
    report=dict(complete=False,success=False,cases=[],scope='Public native interleaved versus experimental Mega; no performance timing',
        frequencies='Read-only GPU1197MHz/EMC2750MHz, six min/max/current readings',
        thermal='>=62C coolto54C, abort>=70C')
    guard=None;before=None
    try:
        sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks')]
        import torch
        import grain_gemm
        import grain_gemm.moe as implementation
        from bench_moe_interleaved import Guard,hash_package
        from local_ctypes_api import debug_moe_ffn_mega,moe_ffn_mega,build_metadata
        torch.set_num_threads(4);torch.cuda.set_device(0)
        if torch.cuda.get_device_capability()!=(11,0):raise RuntimeError('Require Thor SM110')
        guard=Guard(torch);guard.state()
        metadata=build_metadata(args.library)
        report['build']=metadata
        def provenance():
            return dict(experiment={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.cu','.cuh','.hpp','.h')},
                library=sha(metadata['library']),current=hash_package(ROOT/'src/grain_gemm'))
        before=provenance();report['provenance_before']=before
        stream_options=[('default',torch.cuda.current_stream()),('alternate',torch.cuda.Stream())]
        if metadata['silu_lut']:
            import ctypes
            import local_ctypes_api as api
            native,_=api._load_library(metadata['library'])
            first_stream=torch.cuda.Stream();second_stream=torch.cuda.Stream()
            waited=[]
            class WaitLoggingStream:
                cuda_stream=second_stream.cuda_stream
                def wait_event(self,event):
                    waited.append(event)
                    second_stream.wait_event(event)
            verify=native.grain_moe_silu_lut_verify
            verify.argtypes=[ctypes.c_void_p]*3;verify.restype=ctypes.c_int
            original=torch.empty(65536,device='cuda',dtype=torch.float32)
            looked_up=torch.empty_like(original)
            def initialize_across_streams():
                with torch.cuda.stream(first_stream):
                    api._initialize_optional(native,metadata['library'],torch.device('cuda',0),first_stream)
                with torch.cuda.stream(second_stream):
                    api._initialize_optional(native,metadata['library'],torch.device('cuda',0),WaitLoggingStream())
                    status=verify(original.data_ptr(),looked_up.data_ptr(),second_stream.cuda_stream)
                    if status:raise RuntimeError('Two-stream SiLU verification failed: '+str(status))
            guard.call(initialize_across_streams)
            equal=bool(torch.equal(original.view(torch.int32),looked_up.view(torch.int32)))
            if len(waited)!=1 or not equal:
                raise AssertionError('LUT first initialization/event dependency failed')
            report['lut_first_use_stream_dependency']=dict(owner_stream=first_stream.cuda_stream,
                waiting_stream=second_stream.cuda_stream,cached_event_waits=len(waited),
                table_patterns=65536,all_fp32_bits_equal=equal,
                scope='Targeted sequential two-stream first-init/event dependency; not GPU concurrency stress')
        specifications=[('balanced_partial',2,17,2),('duplicate_masked',4,17,3),
            ('skew_empty_experts',4,129,1),('duplicate_partial',4,129,4),
            ('all_masked_invalid',4,17,4),('zero_tokens',2,0,2)]
        with torch.inference_mode():
            for name,e,t,r in specifications:
                guard.cool();torch.manual_seed(70421+e+t+r)
                h,i=1536,1024
                qx=torch.randint(-128,128,(t,h),device='cuda',dtype=torch.int8)
                sx=(torch.randn((t,h//256),device='cuda')*.001).contiguous()
                w1=torch.randint(-128,128,(e,2*i,h),device='cuda',dtype=torch.int8).transpose(-2,-1)
                w2=torch.randint(-128,128,(e,h,i),device='cuda',dtype=torch.int8).transpose(-2,-1)
                s1=(torch.randn((e,h//256,2*i),device='cuda')*.001).contiguous()
                s2=(torch.randn((e,i//256,h),device='cuda')*.001).contiguous()
                sx[:,::3]=0;s1[:,:,::7]=0;s2[:,:,::11]=0
                if t:qx[0,0]=-128
                w1[0,0,0]=-128;w2[0,0,0]=-128
                token=torch.arange(t,device='cuda')
                ids=((token[:,None]*r+torch.arange(r,device='cuda'))%e).long().contiguous()
                scores=torch.rand((t,r),device='cuda');scores=(scores/scores.sum(-1,keepdim=True)).bfloat16().float().contiguous()
                mask=None
                if name.startswith('duplicate'):
                    ids=(token[:,None]%e).expand(-1,r).contiguous()
                if name=='duplicate_masked':
                    mask=(token%3!=1).contiguous();ids[~mask]=-9;sx[~mask]=float('nan')
                elif name=='skew_empty_experts':
                    ids.zero_()
                elif name=='all_masked_invalid':
                    mask=torch.zeros(t,device='cuda',dtype=torch.bool);ids.fill_(e+11);sx.fill_(float('nan'))
                valid_ids=ids if mask is None else ids[mask]
                counts=torch.bincount(valid_ids.flatten(),minlength=e).int()
                keywords=dict(scale_x=sx,scale_gate_up=s1,scale_down=s2,token_mask=mask,
                    output_dtype=torch.bfloat16,gate_up_layout='interleaved',backend='cutlass')
                for consumer_warps in (8,16):
                    for stream_name,stream in stream_options:
                        with torch.cuda.stream(stream):
                            baseline=capture_current(lambda:grain_gemm.moe_ffn(qx,w1,w2,scores,ids,**keywords),guard,implementation)
                            actual=guard.call(lambda:debug_moe_ffn_mega(qx,w1,w2,scores,ids,**keywords,
                                consumer_warps=consumer_warps,library=args.library))
                            checks=exact_checks(actual,baseline,counts,torch)
                            output=guard.call(lambda:moe_ffn_mega(qx,w1,w2,scores,ids,**keywords,
                                consumer_warps=consumer_warps,library=args.library))
                            checks['normal_launch_output_bits_equal']=bool(torch.equal(output.view(torch.int16),baseline['output'].view(torch.int16)))
                            checks['masked_rows_zero']=mask is None or bool((output[~mask]==0).all())
                            if not all(checks.values()):raise AssertionError(checks)
                        report['cases'].append(dict(name=name,experts=e,tokens=t,top_k=r,consumer_warps=consumer_warps,
                            stream=stream_name,capacity=actual['capacity'],checks=checks))
                        print(f'{name} E{e} T{t} R{r} CW{consumer_warps} {stream_name}: PASS',flush=True)
                if t and name=='balanced_partial':
                    bad_ids=ids.clone();bad_ids[0,0]=e
                    rejected={}
                    for method,fn in [('current',grain_gemm.moe_ffn),('mega',moe_ffn_mega)]:
                        try:
                            more=dict(library=args.library) if method=='mega' else {}
                            guard.call(lambda fn=fn,more=more:fn(qx,w1,w2,scores,bad_ids,**keywords,**more))
                        except ValueError:
                            rejected[method]=True
                    if rejected!={'current':True,'mega':True}:raise AssertionError(rejected)
                    report['invalid_live_route_rejected']=rejected
            if metadata['silu_lut']:
                import ctypes
                import local_ctypes_api as api
                path=Path(metadata['library'])
                native,_=api._load_library(str(path))
                stream=torch.cuda.current_stream()
                api._initialize_optional(native,path,torch.device('cuda',0),stream)
                verify=native.grain_moe_silu_lut_verify
                verify.argtypes=[ctypes.c_void_p]*3
                verify.restype=ctypes.c_int
                original=torch.empty(65536,device='cuda',dtype=torch.float32)
                looked_up=torch.empty_like(original)
                def verify_table():
                    status=verify(original.data_ptr(),looked_up.data_ptr(),stream.cuda_stream)
                    if status:raise RuntimeError('SiLU table verification launch failed: '+str(status))
                guard.call(verify_table)
                equal=bool(torch.equal(original.view(torch.int32),looked_up.view(torch.int32)))
                report['silu_lut_exhaustive']=dict(patterns=65536,all_fp32_bits_equal=equal,
                    includes='All BF16 bit patterns: finite values, signed zero/subnormal, infinities and NaNs')
                if not equal:raise AssertionError('Exhaustive SiLU table differs from original expression')
            report.update(complete=True,success=True)
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        print(report['failure']['traceback'],file=sys.stderr,flush=True)
    finally:
        if before is not None:
            try:
                after=provenance();report['provenance_after']=after;report['provenance_unchanged']=after==before
                if after!=before:report.update(complete=False,success=False)
            except Exception as error:report.update(complete=False,success=False,provenance_error=repr(error))
        if guard is not None:
            report['clock_temperature_snapshots']=guard.states
            report['maximum_temperature_c']=max((s['temperature_c'] for s in guard.states),default=None)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'success={report["success"]}; cases={len(report["cases"])}; output={args.output}',flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
