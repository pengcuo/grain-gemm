#!/usr/bin/env python3
"""Portable four-way Mega INT8 / Mega FP8 / Mega BF16 / Cosmos BF16 comparison.

No GPU work occurs on import. Run with exclusive ownership of the Thor GPU.
This is a portable reproduction of the archived measurement harness; the
archived numerical report was measured with its separately frozen executable.

Candidate adapter contract:
  make_experiment(context, **candidate_kwargs) -> experiment
  experiment.launch_full() -> BF16[T,H]
  experiment.debug() -> {output,qh,sh,expert_y,counts[,capacity]}
Optional experiment.metadata is JSON-serializable. The timed launch must include
routing preparation, per-call workspaces, both projections/activation/quant and
combine. Cached compilation, weights and DeepInfer reusable Mega workspace are
outside timing for all methods. Context exposes torch,guard,qx,sx,w_gate_up,
scale_gate_up,w_down,scale_down,ids,scores,token_mask,baseline_debug,baseline,
current, kernels, tokens,experts,hidden,intermediate,top_k,output_dtype.

Full-call timings are eager CUDA events, never CUDA Graphs. A 256MB flush is
outside every timed sample. Clocks are read only (all six fields); >=62C cool to
54C, abort>=70C. Default 4 methods/4 rounds/3 samples rotate and reverse order.
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HELPER=HERE/'compat.py'
PAIRED=ROOT/'benchmarks/bench_moe_interleaved.py'
FAST=HERE/'fp8_input.py'
METHODS=('mega_int8','fp8_mega','bf16_mega','cosmos_bf16_compiled')
BF16_METHODS=('bf16_mega','cosmos_bf16_compiled')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkout_state(directory):
    """Git metadata is optional when running from an extracted source archive."""
    try:
        return dict(head=subprocess.check_output(
            ['git','-C',str(directory),'rev-parse','HEAD'],text=True,
            stderr=subprocess.DEVNULL).strip(),status=subprocess.check_output(
            ['git','-C',str(directory),'status','--short'],text=True,
            stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        return dict(head=None,status=None)


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@contextmanager
def nvtx(torch,label):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def order_for(round_index,sample):
    offset=(round_index+sample)%len(METHODS)
    order=METHODS[offset:]+METHODS[:offset]
    return order if sample%2==0 else order[::-1]


def save(report,path):
    report['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
    fields=['tokens','method','median_ms','effective_tops_or_tflops','unit','round_medians_ms']
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for case in report['cases']:
            for method,result in case.get('results',{}).items():
                writer.writerow(dict(tokens=case['tokens'],method=method,**{key:result[key] for key in fields[2:]}))


def metadata_payload_hashes(metadata):
    """Fingerprint declared external code/native payloads, including variants."""
    paths=set()
    def visit(value):
        if isinstance(value,dict):
            for item in value.values():
                visit(item)
        elif isinstance(value,(list,tuple)):
            for item in value:
                visit(item)
        elif isinstance(value,str) and value.startswith('/'):
            path=Path(value)
            if path.suffix in ('.so','.cu','.cuh','.h','.hpp','.cpp','.py') and path.is_file():
                paths.add(path.resolve())
    visit(metadata)
    return {str(path):sha(path) for path in sorted(paths)}


def capture_public(package,launch,guard):
    """Untimed debug capture at the actual interleaved public G2 boundary."""
    implementation=importlib.import_module(package.__name__+'.moe')
    native=implementation._native_gemm
    record=dict(native_calls=[])
    def wrapped(*args,**kwargs):
        result=native(*args,**kwargs)
        record['native_calls'].append(dict(a_shape=list(args[0].shape),weight_shape=list(args[1].shape),backend=args[5]))
        record.update(qh=args[0].clone(),sh=args[2].clone(),expert_y=args[4].clone(),counts=kwargs['counts'].clone())
        return result
    with patch.object(implementation,'_native_gemm',wrapped):
        record['output']=guard.call(launch)
    if len(record['native_calls'])!=1:
        raise AssertionError('Expected one native G2 boundary: '+str(record['native_calls']))
    record['capacity']=record['qh'].shape[1]
    return record


def check_debug(data,reference,torch):
    required=('output','qh','sh','expert_y','counts')
    missing=[name for name in required if name not in data]
    if missing:
        raise AssertionError('Candidate debug() missing '+str(missing))
    checks={}
    for name in required:
        actual,expected=data[name],reference[name]
        if tuple(actual.shape)!=tuple(expected.shape):
            raise AssertionError(f'{name}: candidate shape {tuple(actual.shape)} != baseline {tuple(expected.shape)}')
        view=torch.int32 if name=='sh' else torch.int16 if name in ('output','expert_y') else None
        # Unused expert-output padding is uninitialized and is not part of the
        # combine input contract. Compare every useful expert row, not garbage.
        if name=='expert_y':
            valid=torch.arange(actual.shape[1],device=actual.device)[None,:]<reference['counts'][:,None]
            actual,expected=actual[valid],expected[valid]
        checks[name+'_bits_equal']=bool(torch.equal(actual.view(view) if view else actual,expected.view(view) if view else expected))
    qh,sh=data['qh'],data['sh']
    padding=torch.arange(qh.shape[1],device=qh.device)[None,:]>=reference['counts'][:,None]
    zero_sf=(torch.tensor(1.e-12,dtype=torch.float32)/torch.tensor(127.,dtype=torch.float32)).to(qh.device)
    checks['padding_q_zero']=bool((qh[padding]==0).all())
    checks['padding_scale_bits_equal']=bool((sh[padding].view(torch.int32)==zero_sf.view(torch.int32)).all())
    checks['all_finite']=bool(torch.isfinite(data['output']).all())
    if not all(checks.values()):
        raise AssertionError('Candidate intermediate/full-output equivalence failed: '+str(checks))
    return checks


class PublicExperiment:
    """Control/default adapter; permits testing the harness before new builds."""
    metadata={'scope':'full public interleaved moe_ffn(auto)','adapter':'public'}
    def __init__(self,context):
        self.context=context
    def launch_full(self):
        c=self.context
        return c.current.moe_ffn(c.qx,c.w_gate_up,c.w_down,c.scores,c.ids,
            scale_x=c.sx,scale_gate_up=c.scale_gate_up,scale_down=c.scale_down,
            token_mask=c.token_mask,gate_up_layout='interleaved',backend='auto')
    def debug(self):
        return capture_public(self.context.current,self.launch_full,self.context.guard)



def full_bf16_expert_reference(x,ids,scores,weights,torch,guard):
    """Untimed full-output independent BF16 projection reference.

    Uses ordinary per-expert torch.mm, not grouped_mm or any Mega kernel.
    Gate/Up each round to BF16, the FP32 SiLU*up*score product rounds once to
    BF16 before Down, Down rounds to BF16, then FP32 slot-order combination.
    This honors score-before-Down. Different GEMM and scatter rounding is
    checked with the explicit bounds recorded in the report, not bit equality.
    """
    tokens,hidden=x.shape;top_k=ids.shape[1]
    slots=guard.call(lambda:torch.empty((tokens,top_k,hidden),device=x.device,dtype=torch.bfloat16))
    flat_ids=ids.reshape(-1);flat_scores=scores.reshape(-1)
    for expert in range(weights[0].shape[0]):
        positions=guard.call(lambda expert=expert:torch.nonzero(flat_ids==expert,as_tuple=False).flatten())
        if not positions.numel():continue
        def expert_projection(positions=positions,expert=expert):
            rows=positions//top_k
            gate_up=torch.mm(x.index_select(0,rows),weights[0][expert].t())
            gate,up=gate_up.chunk(2,-1)
            hidden_bf16=(torch.nn.functional.silu(gate.float())*up.float()*flat_scores.index_select(0,positions)[:,None]).bfloat16()
            expert_y=torch.mm(hidden_bf16,weights[1][expert].t())
            slots.view(-1,hidden).index_copy_(0,positions,expert_y)
        guard.call(expert_projection)
    def combine():
        output=torch.zeros((tokens,hidden),device=x.device,dtype=torch.float32)
        for slot in range(top_k):output.add_(slots[:,slot].float())
        return output.bfloat16()
    return guard.call(combine)


def full_difference(value,reference,torch,guard):
    """All elements participate; reduction scalars alone cross to the CPU."""
    def compare():
        a=value.float();b=reference.float();delta=a-b
        return dict(relative_l2=float(delta.norm()/b.norm().clamp_min(1e-12)),
                    max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()),
                    all_finite=bool(torch.isfinite(a).all()))
    return guard.call(compare)


def check_full_bf16(value,reference,torch,guard):
    metrics=full_difference(value,reference,torch,guard)
    if not metrics['all_finite'] or metrics['relative_l2']>.03 or metrics['max_abs']>.05:
        raise AssertionError('Full-output independent BF16 reference failed: '+str(metrics))
    guard.call(lambda:torch.testing.assert_close(value,reference,rtol=.025,atol=.008))
    metrics.update(assert_close_rtol=.025,assert_close_atol=.008,relative_l2_limit=.03,max_abs_limit=.05,
                   checked_elements=value.numel(),full_output_close=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',nargs='+',type=int,default=[4096,8192,16384,32768])
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True,help='Output JSON file; CSV is written beside it')
    parser.add_argument('--before-package-dir',type=Path,help='Optional frozen public grain_gemm package for untimed exact INT8 stage checks; defaults to the current public package')
    parser.add_argument('--int8-library',type=Path,required=True,help='Selected exact-SiLU-LUT interleaved Mega INT8 .so')
    parser.add_argument('--deepinfer-dir',type=Path,required=True,help='Clean pinned DeepInfer checkout containing vendor/DeepGEMM')
    parser.add_argument('--build-deps-dir',type=Path,help='Optional additional Python dependency directory; normal environment packages are otherwise used')
    parser.add_argument('--candidate-module',default=str(ROOT/'experiments/thor_interleaved_mega/candidate.py'),help='Selected INT8 Mega adapter Python file or import name')
    parser.add_argument('--candidate-factory',default='make_experiment')
    parser.add_argument('--candidate-kwargs',default='{}',help='JSON object passed to candidate factory')
    parser.add_argument('--bf16-candidate-module',default=str(HERE/'candidate.py'))
    parser.add_argument('--bf16-candidate-kwargs',default='{}')
    parser.add_argument('--deepgemm-package-dir',type=Path,required=True,help='Path to isolated built deep_gemm package, not its parent')
    parser.add_argument('--deepgemm-source-dir',type=Path,required=True)
    parser.add_argument('--cosmos-dir',type=Path,required=True)
    parser.add_argument('--skip-cpu-reference',action='store_true',help='Explicitly omit only two-token CPU references; full independent BF16 output checks still run')
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if min(args.tokens)<2 or len(set(args.tokens))!=len(args.tokens) or min(args.rounds,args.samples)<1 or args.warmup<0 or (args.smoke_only and args.profile_only):
        parser.error('Require unique tokens>=2, positive rounds/samples, nonnegative warmup and one mode')
    if not args.candidate_module or not args.bf16_candidate_module:
        parser.error('Both measured candidate adapters must be supplied')
    candidate_kwargs=json.loads(args.candidate_kwargs)
    if not isinstance(candidate_kwargs,dict):
        parser.error('--candidate-kwargs must decode to an object')
    if 'library' in candidate_kwargs and Path(candidate_kwargs['library']).resolve()!=args.int8_library.resolve():
        parser.error('--int8-library and candidate kwargs library disagree')
    candidate_kwargs['library']=str(args.int8_library.resolve())
    if args.output.suffix!='.json':
        parser.error('--output must end in .json')
    original=(args.deepinfer_dir.resolve()/'vendor/DeepGEMM')
    before_package=(args.before_package_dir or ROOT/'src/grain_gemm').resolve()
    bf16_candidate_kwargs=json.loads(args.bf16_candidate_kwargs)
    if not isinstance(bf16_candidate_kwargs,dict):
        parser.error('--bf16-candidate-kwargs must decode to an object')
    package=args.deepgemm_package_dir.resolve();source=args.deepgemm_source_dir.resolve()
    output=args.output.resolve()
    report=dict(complete=False,success=False,stage='initialize',cases=[],calls=[],
        script_path=str(Path(__file__).resolve()),script_sha256=sha(__file__),seed=2026,
        experts=32,hidden=1536,intermediate=1024,top_k=4,tokens=args.tokens,
        before_package=str(before_package),deepinfer_dir=str(args.deepinfer_dir.resolve()),
        ported_harness=True,archived_measurement_executable=False,candidate_module=args.candidate_module,candidate_kwargs=candidate_kwargs,
        bf16_candidate_module=args.bf16_candidate_module,bf16_candidate_kwargs=bf16_candidate_kwargs,
        cpu_reference_enabled=not args.skip_cpu_reference,
        rounds=args.rounds,samples_per_round=args.samples,warmup=args.warmup,
        timing='Full eager CUDA-event span, no graphs. Routing/per-call allocations, two projections, SwiGLU/quant and combine included. Input/weight quantization, weight transforms, compilation and reusable Mega buffer setup excluded.',
        scopes={'bf16_mega':'Resident immutable transformed BF16 weights and reusable Mega workspace; input/route/score copies, output allocation and complete BF16 Mega included',
                'cosmos_bf16_compiled':'Original Cosmos counts,stable sorting/padding/gather,two torch._grouped_mm,SwiGLU+score before Down,BF16 scatter-add; torch.compile(fullgraph=True,dynamic=True), compilation excluded',
                'mega_int8':'Candidate full launch; inspect candidate metadata for stages and reusable workspace policy',
                'fp8_mega':'Resident original FP8 input/scales; ID/score copies, Y allocation and original guard-only SM110 Mega dispatch+G1+activation/quant+G2+combine'},
        formats={'int8':'INT8 G256, FP32 scales; BF16 gate/up and G2 rounding; score after G2',
                 'fp8':'FP8 E4M3 G32 UE8M0; fast_math=True; score before hidden quantization',
                 'bf16_mega':'BF16 source activations/weights; BF16 gate/up,FP32SwiGLU*score then one hidden BF16round before Down; fast_math=True,clamp=None',
                 'cosmos_bf16_compiled':'BF16 source activations/weights; fused SwiGLU*score before Down,BF16 scatter-add'},
        input_policy='Identical BF16 source input/source weights/route IDs/scores; quantized independently to each format outside timing',
        numerator='6*T*R*H*I useful operations excluding padding and non-GEMM arithmetic; INT8 TOPS / FP8 and BF16 TFLOPS',
        clock_policy='Read only: GPU min/max/current1197MHz, EMC min/max/current2750MHz; nominal GPU reduced to1.2GHz',
        thermal='>=62C coolto54C, abort>=70C; short calls only;256MB L2 flush outside each timed sample',
        precision='Full INT8 QH/SH/useful BF16 expertY/output bits versus the public reference package; FP8 input/repeated full bits; both BF16 methods finite/full-output error versus independent perexpert torch.mm reference plus optional two-token CPU references. No cross-format or BF16 GEMM bitexact claim.',
        bf16_reference=dict(scope='Full independent perexpert ordinary torch.mm, not grouped_mm/Mega, GateUpBF16, one FP32SwiGLU*Up*score-to-BF16 rounding before Down, DownBF16 and FP32 serial slot combine',assert_close_rtol=.025,assert_close_atol=.008,relative_l2_limit=.03,max_abs_limit=.05),
        fp8_baseline='Explicit SM110 host-guard compatibility experiment with original kernel arithmetic/layout/schedule; no communications on one rank',
        fp8_expert_stats='Cumulative counters initialized once per case; compare counts after first call only; later calls preserve original behavior',
        bf16_expert_stats='Same cumulative-counter policy as FP8; check full input/ID/score copies and all expert counts immediately after first BF16 call')
    save(report,output)
    provenance_before=None;guard=None;prepared=[];profiler_started=False;candidate_payloads_before={}
    try:
        sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks'),str(package.parent),str(HERE)]
        if args.build_deps_dir is not None:
            sys.path.insert(0,str(args.build_deps_dir.resolve()))
        paired=load_file('_interleaved_mega_paired',PAIRED)
        helper=load_file('_interleaved_mega_fp8_helper',HELPER)
        import torch
        import grain_gemm as current
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8,unpack_ue8m0_from_int
        from fp8_input import quantize_input_g32_into
        from grain_gemm.kernels import moe as kernels
        from cosmos_moe_baseline import load_cosmos_experts,make_cosmos_forward,_source_metadata
        from bench_moe_cosmos import bf16_cpu_reference
        baseline=(current if before_package==Path(current.__file__).resolve().parent
            else paired.load_package(before_package,'grain_gemm_before_mega_benchmark'))
        if Path(current.__file__).resolve().parent!=ROOT/'src/grain_gemm' or Path(deep_gemm.__file__).resolve().parent!=package:
            raise RuntimeError('Unexpected current or DeepGEMM package path')
        candidate_module=None
        if args.candidate_module:
            candidate_module=(load_file('_interleaved_mega_candidate',Path(args.candidate_module))
                if args.candidate_module.endswith('.py') else importlib.import_module(args.candidate_module))
        bf16_candidate_module=(load_file('_mega_bf16_candidate',Path(args.bf16_candidate_module))
            if args.bf16_candidate_module.endswith('.py') else importlib.import_module(args.bf16_candidate_module))
        def source_state():
            result=helper.source_state(source,original)
            result['guard_edit_count']=sum(len(v) for v in helper.COMPAT_HOST_GUARDS.values())
            return result
        script_paths=[Path(__file__).resolve(),PAIRED,HELPER,FAST,Path(bf16_candidate_module.__file__).resolve(),
            ROOT/'benchmarks/cosmos_moe_baseline.py',ROOT/'benchmarks/bench_moe_cosmos.py',ROOT/'benchmarks/bench_moe_ffn.py']
        if candidate_module is not None:
            script_paths.append(Path(candidate_module.__file__).resolve())
        def provenance():
            return dict(current=paired.hash_package(ROOT/'src/grain_gemm'),baseline=paired.hash_package(before_package),
                deepinfer_source=helper.verify_deepinfer(args.deepinfer_dir),cosmos_source=_source_metadata(args.cosmos_dir),
                deepgemm_source=source_state(),deepgemm_payload=helper.package_payload_state(package,source),
                deepgemm_python=helper.package_python_state(package,source/'deep_gemm'),
                scripts={str(path):sha(path) for path in script_paths},deepgemm_native_sha256=sha(deep_gemm._C.__file__))
        provenance_before=provenance();report['provenance_before']=provenance_before
        torch.set_num_threads(4);torch.cuda.set_device(0)
        if torch.cuda.get_device_capability()!=(11,0):
            raise RuntimeError('Require Thor SM110')
        guard=paired.Guard(torch);report['all_clock_temperature_snapshots']=guard.states
        report['state_before']=guard.state()
        report['software']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
        report['checkout']=checkout_state(ROOT)
        deep_gemm.use_deterministic_algorithms(True)
        with torch.inference_mode():
            report['stage']='prepare_weights';save(report,output)
            torch.manual_seed(2026)
            e,h,i,r=32,1536,1024,4
            weights=[guard.call(lambda:(torch.randn((e,2*i,h),device='cuda',dtype=torch.bfloat16)/h**.5).contiguous()),
                     guard.call(lambda:(torch.randn((e,h,i),device='cuda',dtype=torch.bfloat16)/i**.5).contiguous())]
            q1,s1=guard.call(lambda:paired.prepare_weight(weights[0],torch))
            q2,s2=guard.call(lambda:paired.prepare_weight(weights[1],torch))
            qi,si=guard.call(lambda:paired.interleave_weight(q1,s1))
            raw_fp8,required=[],[]
            for weight in weights:
                guard.cool()
                q=torch.empty_like(weight,dtype=torch.float8_e4m3fn)
                sf=torch.empty((e,weight.shape[1],weight.shape[2]//32),device='cuda',dtype=torch.float32)
                for expert in range(e):
                    q[expert],sf[expert]=per_token_cast_to_fp8(weight[expert],use_ue8m0=True,gran_k=32)
                raw_fp8.append((q,sf))
                required.append((q,deep_gemm.transform_sf_into_required_layout(sf,weight.shape[1],weight.shape[2],(1,32),e)))
                torch.cuda.synchronize();guard.state()
            transformed=guard.call(lambda:deep_gemm.transform_weights_for_mega_moe(*required))
            cosmos_module,cosmos_metadata=guard.call(lambda:load_cosmos_experts(args.cosmos_dir,e,h,i,r,'cuda',torch.bfloat16,*weights))
            cosmos=torch.compile(make_cosmos_forward(cosmos_module),fullgraph=True,dynamic=True)
            report['cosmos']=cosmos_metadata
            report['cosmos_compilation']=dict(fullgraph=True,dynamic=True,backend='inductor',mode='default')
            for tokens in args.tokens:
                report['stage']=f'T{tokens}:setup';save(report,output);guard.cool()
                torch.manual_seed(2026+tokens)
                x=guard.call(lambda:torch.randn((tokens,h),device='cuda',dtype=torch.bfloat16))
                ids=guard.call(lambda:torch.randn((tokens,e),device='cuda').topk(r,dim=-1).indices.contiguous())
                scores=guard.call(lambda:torch.rand((tokens,r),device='cuda'))
                scores=(scores/scores.sum(-1,keepdim=True)).bfloat16().float().contiguous()
                qx,sx=guard.call(lambda:kernels.quantize_rows(x))
                def baseline_forward(qx=qx,sx=sx,ids=ids,scores=scores):
                    return baseline.moe_ffn(qx,qi,q2,scores,ids,scale_x=sx,scale_gate_up=si,
                        scale_down=s2,gate_up_layout='interleaved',backend='auto')
                report['stage']=f'T{tokens}:capture_baseline';save(report,output)
                reference=capture_public(baseline,baseline_forward,guard)
                context=SimpleNamespace(torch=torch,guard=guard,qx=qx,sx=sx,w_gate_up=qi,scale_gate_up=si,
                    w_down=q2,scale_down=s2,ids=ids,scores=scores,token_mask=None,baseline_debug=reference,
                    baseline=baseline,current=current,kernels=kernels,tokens=tokens,experts=e,hidden=h,
                    intermediate=i,top_k=r,output_dtype=torch.bfloat16,
                    x=x,x_bf16=x,weights_linear_gate_up=weights[0],weights_linear_down=weights[1],
                    weights=weights,scores_bf16=scores.bfloat16(),deep_gemm=deep_gemm,helper=helper,
                    deepgemm_package_dir=package,deepgemm_source_dir=source)
                experiment=guard.call(lambda:getattr(candidate_module,args.candidate_factory)(context,**candidate_kwargs)) if candidate_module else PublicExperiment(context)
                if not callable(getattr(experiment,'launch_full',None)) or not callable(getattr(experiment,'debug',None)):
                    raise TypeError('Candidate requires launch_full() and debug()')
                bf16_experiment=guard.call(lambda:bf16_candidate_module.make_experiment(context,**bf16_candidate_kwargs))
                if not callable(getattr(bf16_experiment,'launch_full',None)):
                    raise TypeError('BF16 candidate requires launch_full()')
                prepared.append(dict(buffer=None,bf16_experiment=bf16_experiment))
                bf16_metadata=getattr(bf16_experiment,'metadata',{})
                candidate_payloads=metadata_payload_hashes({'int8':getattr(experiment,'metadata',{}),'bf16':bf16_metadata})
                for payload,digest in candidate_payloads.items():
                    if payload in candidate_payloads_before and candidate_payloads_before[payload]!=digest:
                        raise RuntimeError('Candidate native/source changed between cases: '+payload)
                    candidate_payloads_before[payload]=digest
                report['candidate_payloads_before']=candidate_payloads_before
                case=dict(tokens=tokens,capacity=reference['capacity'],expert_counts=reference['counts'].cpu().tolist(),
                    candidate_payloads=candidate_payloads,
                    candidate_metadata=getattr(experiment,'metadata',{}),bf16_candidate_metadata=bf16_metadata,results={},rounds=[],checks={})
                report['cases'].append(case)
                buffer=guard.call(lambda:deep_gemm.get_symm_buffer_for_mega_moe(helper.SingleCardGroup(),e,tokens,r,h,i,num_shared_experts=0,mma_type='fp8xfp8'))
                prepared[-1]['buffer']=buffer
                guard.call(lambda:quantize_input_g32_into(x,buffer.x,buffer.x_sf,rows_per_program=16))
                stats=torch.zeros(e,device='cuda',dtype=torch.int32)
                def fp8_forward(buffer=buffer,stats=stats,ids=ids,scores=scores,tokens=tokens):
                    with nvtx(torch,'stage:fp8_mega:input_copy') if args.profile_only else nullcontext():
                        buffer.topk_idx[:tokens].copy_(ids);buffer.topk_weights[:tokens].copy_(scores)
                    y=torch.empty((tokens,h),device='cuda',dtype=torch.bfloat16)
                    with nvtx(torch,'stage:fp8_mega:mega') if args.profile_only else nullcontext():
                        deep_gemm.fp8_fp4_mega_moe(y,*transformed,buffer,cumulative_local_expert_recv_stats=stats,activation_clamp=None,fast_math=True)
                    return y
                functions=dict(mega_int8=experiment.launch_full,fp8_mega=fp8_forward,bf16_mega=bf16_experiment.launch_full,
                    cosmos_bf16_compiled=lambda x=x,ids=ids,scores_bf16=context.scores_bf16:cosmos(x,ids,scores_bf16))
                report['stage']=f'T{tokens}:correctness';save(report,output)
                candidate_debug=guard.call(experiment.debug)
                case['checks']['candidate_debug']=check_debug(candidate_debug,reference,torch)
                first_fp8=guard.call(fp8_forward)
                counts=torch.bincount(ids.flatten(),minlength=e).int()
                if not torch.equal(stats,counts) or not torch.equal(counts,reference['counts']):
                    raise AssertionError('Expert route counts differ')
                case['checks']['fp8_first_call_expert_counts_equal']=True
                qx_fp8,sx_fp8=guard.call(lambda:per_token_cast_to_fp8(x,use_ue8m0=True,gran_k=32,use_packed_ue8m0=True))
                case['checks']['fp8_full_input_q_bits_equal']=bool(torch.equal(buffer.x[:tokens].view(torch.uint8),qx_fp8.view(torch.uint8)))
                case['checks']['fp8_full_input_sf_bits_equal']=bool(torch.equal(buffer.x_sf[:tokens],sx_fp8))
                if not all(case['checks'][name] for name in ('fp8_full_input_q_bits_equal','fp8_full_input_sf_bits_equal')):
                    raise AssertionError('FP8 fast producer differs from original cast bits')
                ref_full_bf16=full_bf16_expert_reference(x,ids,scores,weights,torch,guard)
                if not bool(torch.isfinite(ref_full_bf16).all()):
                    raise AssertionError('Independent full BF16 reference is nonfinite')
                first_bf16=guard.call(functions['bf16_mega'])
                def verify_bf16_first_call():
                    bf=bf16_experiment.buffer
                    return dict(first_call_expert_counts_equal=bool(torch.equal(bf16_experiment.stats,counts)),
                        full_input_bits_equal=bool(torch.equal(bf.x[:tokens].view(torch.int16),x.view(torch.int16))),
                        full_route_id_bits_equal=bool(torch.equal(bf.topk_idx[:tokens],ids)),
                        full_route_score_bits_equal=bool(torch.equal(bf.topk_weights[:tokens].view(torch.int32),scores.view(torch.int32))))
                case['checks']['bf16_first_call']=guard.call(verify_bf16_first_call)
                if not all(case['checks']['bf16_first_call'].values()):
                    raise AssertionError('BF16 Mega first-call input/route/count checks failed: '+str(case['checks']['bf16_first_call']))
                first_cosmos=guard.call(functions['cosmos_bf16_compiled'])
                case['checks']['bf16_full_reference']={method:check_full_bf16(value,ref_full_bf16,torch,guard)
                    for method,value in [('bf16_mega',first_bf16),('cosmos_bf16_compiled',first_cosmos)]}
                case['checks']['bf16_mega_vs_cosmos']=full_difference(first_bf16,first_cosmos,torch,guard)
                refs={};ref_common=None
                if not args.skip_cpu_reference:
                    ref_int8=paired.cpu_reference(qx[:2],ids[:2],scores[:2],q1,q2,s1,s2,sx[:2],torch)
                    ref_fp8=helper.cpu_reference(qx_fp8,sx_fp8,ids,scores,raw_fp8,torch,per_token_cast_to_fp8,unpack_ue8m0_from_int,None)
                    ref_bf16=bf16_cpu_reference(x[:2],ids[:2],scores[:2],*weights,torch,fused_activation=True)
                    torch.testing.assert_close(ref_full_bf16[:2].cpu(),ref_bf16,rtol=.025,atol=.008)
                    ref_common=helper.common_fp32_reference(x[:2],ids[:2],scores[:2],weights,torch)
                    refs=dict(mega_int8=ref_int8,fp8_mega=ref_fp8,bf16_mega=ref_bf16,cosmos_bf16_compiled=ref_bf16)
                first=dict(mega_int8=candidate_debug['output'],fp8_mega=first_fp8,bf16_mega=first_bf16,cosmos_bf16_compiled=first_cosmos)
                case['checks']['quantized_reference']={};case['checks']['accuracy_vs_common_fp32']={};case['checks']['repeated_full_output_bits_equal']={}
                for method in METHODS:
                    value=guard.call(functions[method])
                    equal=bool(torch.equal(value.view(torch.int16),first[method].view(torch.int16)))
                    finite=bool(torch.isfinite(value).all())
                    if not finite or (method not in BF16_METHODS and not equal):
                        raise AssertionError(f'{method} repeated finite/exact check failed: {finite} {equal}')
                    if method in BF16_METHODS:
                        check_full_bf16(value,ref_full_bf16,torch,guard)
                        guard.call(lambda value=value,method=method:torch.testing.assert_close(value,first[method],rtol=.025,atol=.008))
                    case['checks']['repeated_full_output_bits_equal'][method]=equal
                    if refs:
                        diff=paired.difference(value[:2],refs[method],torch)
                        if diff['relative_l2']>.03 or diff['max_abs']>.05:
                            raise AssertionError(f'{method} independent CPU reference failed: {diff}')
                        case['checks']['quantized_reference'][method]=diff
                        case['checks']['accuracy_vs_common_fp32'][method]=paired.difference(value[:2],ref_common,torch)
                prepared[-1].update(case=case,functions=functions,first=first,refs=refs,experiment=experiment,reference=reference,ref_full_bf16=ref_full_bf16)
                for _ in range(args.warmup):
                    for method in METHODS:
                        guard.call(functions[method])
                save(report,output)
                if args.smoke_only or args.profile_only:
                    continue
                report['stage']=f'T{tokens}:benchmark';save(report,output)
                results=guard.measure(functions,rounds=args.rounds,samples=args.samples,graph_calls=0)
                for method,result in results.items():
                    result['effective_tops_or_tflops']=6*tokens*r*h*i/(result['median_ms']*1e9)
                    result['unit']='TOPS' if method=='mega_int8' else 'TFLOPS'
                case['results']=results
                case['checks_after_timing']={}
                for method in METHODS:
                    value=guard.call(functions[method])
                    equal=bool(torch.equal(value.view(torch.int16),first[method].view(torch.int16)))
                    finite=bool(torch.isfinite(value).all())
                    checks=dict(all_finite=finite,full_output_bits_equal=equal)
                    if not finite or (method not in BF16_METHODS and not equal):
                        raise AssertionError(f'{method} post-timing finite/exact check failed')
                    if method in BF16_METHODS:
                        checks['full_bf16_reference']=check_full_bf16(value,ref_full_bf16,torch,guard)
                        guard.call(lambda value=value,method=method:torch.testing.assert_close(value,first[method],rtol=.025,atol=.008))
                    if refs:
                        diff=paired.difference(value[:2],refs[method],torch);checks['own_quantized_reference']=diff
                        if diff['relative_l2']>.03 or diff['max_abs']>.05:
                            raise AssertionError(f'{method} post-timing CPU reference failed: {diff}')
                    case['checks_after_timing'][method]=checks
                case['checks_after_timing']['candidate_debug']=check_debug(guard.call(experiment.debug),reference,torch)
                print(f'T{tokens}: '+', '.join(f'{method}={results[method]["median_ms"]:.6f}ms' for method in METHODS),flush=True)
                buffer.destroy()
                if callable(getattr(bf16_experiment,'close',None)):bf16_experiment.close()
                prepared[-1].clear();save(report,output)
            if args.profile_only:
                report['stage']='profile_capture';save(report,output)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();b.record();b.synchronize()
                torch.cuda.profiler.start();profiler_started=True
                for data in prepared:
                    case=data['case'];tokens=case['tokens']
                    for rd in range(4):
                        for method in order_for(rd,0):
                            guard.cool();guard.flush.zero_();torch.cuda.synchronize();before_state=guard.state()
                            label=f'profile_interleaved_mega:T{tokens}:{method}:r{rd}'
                            with nvtx(torch,label):
                                a.record();value=data['functions'][method]();b.record();b.synchronize()
                            record=dict(label=label,tokens=tokens,method=method,round=rd,event_elapsed_ms=a.elapsed_time(b),before=before_state,after=guard.state())
                            equal=bool(torch.equal(value.view(torch.int16),data['first'][method].view(torch.int16)))
                            record['checks']=dict(full_output_bits_equal=equal,all_finite=bool(torch.isfinite(value).all()))
                            report['calls'].append(record)
                            if method in BF16_METHODS:
                                record['checks']['full_bf16_reference']=check_full_bf16(value,data['ref_full_bf16'],torch,guard)
                            if not record['checks']['all_finite'] or (method not in BF16_METHODS and not equal):
                                raise AssertionError('Profile correctness changed: '+label)
                torch.cuda.synchronize();torch.cuda.profiler.stop();profiler_started=False
            report.update(complete=True,success=True,stage='finished')
    except Exception as error:
        report['failure']=dict(stage=report['stage'],type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        print(f'FAILED {report["stage"]}: {error}',file=sys.stderr,flush=True)
    finally:
        if profiler_started:
            try:
                torch.cuda.synchronize();torch.cuda.profiler.stop()
            except Exception as error:
                report['profiler_stop_error']=repr(error)
        for data in prepared:
            bf16_exp=data.get('bf16_experiment')
            if bf16_exp is not None and callable(getattr(bf16_exp,'close',None)):
                try:bf16_exp.close()
                except Exception as error:report.setdefault('cleanup_errors',[]).append(repr(error))
            if data.get('buffer') is not None:
                try:
                    data['buffer'].destroy()
                except Exception as error:
                    report.setdefault('cleanup_errors',[]).append(repr(error))
        if candidate_payloads_before:
            try:
                payloads_after={path:sha(path) for path in candidate_payloads_before}
                report['candidate_payloads_after']=payloads_after
                report['candidate_payloads_unchanged']=payloads_after==candidate_payloads_before
                if payloads_after!=candidate_payloads_before:
                    report.update(complete=False,success=False,candidate_payload_error='External candidate native/source bytes changed during run')
            except Exception as error:
                report.update(complete=False,success=False,candidate_payload_error=repr(error))
        if provenance_before is not None:
            try:
                after=provenance();report['provenance_after']=after;report['provenance_unchanged']=after==provenance_before
                if after!=provenance_before:
                    report.update(complete=False,success=False,provenance_error='Source/native/script bytes changed during run')
            except Exception as error:
                report.update(complete=False,success=False,provenance_error=repr(error))
        if guard is not None:
            report['all_clock_temperature_snapshots']=guard.states
            report['maximum_temperature_c']=max((v['temperature_c'] for v in guard.states),default=None)
        save(report,output)
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
