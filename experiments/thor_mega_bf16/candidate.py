"""Isolated exact-SM110 host-guard BF16 Mega comparison adapter."""
import hashlib
from pathlib import Path

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Experiment:
    def __init__(self, context):
        c, torch = context, context.torch
        self.c = c
        self.dg = context.deep_gemm
        package=Path(c.deepgemm_package_dir).resolve()
        source=Path(c.deepgemm_source_dir).resolve()
        if Path(self.dg.__file__).resolve().parent != package:
            raise RuntimeError('Import the isolated BF16-enabled shared build: '+str(package))
        if (c.experts, c.hidden, c.intermediate, c.top_k) != (32, 1536, 1024, 4):
            raise ValueError('This comparison targets E32/H1536/I1024/Top4')
        tensors = [c.x, c.weights_linear_gate_up, c.weights_linear_down, c.ids, c.scores]
        shapes = [(c.tokens,c.hidden), (c.experts,2*c.intermediate,c.hidden),
                  (c.experts,c.hidden,c.intermediate), (c.tokens,c.top_k), (c.tokens,c.top_k)]
        dtypes = [torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int64,torch.float32]
        for value, shape, dtype in zip(tensors, shapes, dtypes):
            if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f'Require contiguous {shape} / {dtype}')
            if value.device != c.x.device or value.device.type != 'cuda':
                raise ValueError('All comparison tensors must share a CUDA device')
        if c.output_dtype != torch.bfloat16 or torch.cuda.get_device_capability(c.x.device) != (11,0):
            raise ValueError('Require Thor SM110 and BF16 output')
        group = c.helper.SingleCardGroup()
        self.buffer = c.guard.call(lambda: self.dg.get_symm_buffer_for_mega_moe(
            group, c.experts, c.tokens, c.top_k, c.hidden, c.intermediate,
            num_shared_experts=0, mma_type='bf16xbf16', activation='swiglu'))
        self.w1, self.w2 = c.guard.call(lambda: self.dg.transform_weights_for_mega_moe(
            c.weights_linear_gate_up, c.weights_linear_down, activation='swiglu'))
        self.stats = c.guard.call(lambda: torch.zeros(c.experts,device=c.x.device,dtype=torch.int32))
        native = Path(self.dg._C.__file__).resolve()
        host = source/'csrc/apis/mega_moe.hpp'
        kernel = source/'deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh'
        self.metadata = dict(adapter=str(Path(__file__).resolve()), source=str(kernel),
            host_source=str(host), native=str(native), python_api=str(source/'deep_gemm/mega/__init__.py'),
            host_sha256=_sha(host), kernel_sha256=_sha(kernel), native_sha256=_sha(native),
            scope='BF16 input/ID/score copies into reusable single-rank SymmBuffer, BF16 Y allocation, original BF16 Mega dispatch/G1/SwiGLU/G2/combine',
            preparation='Immutable gran8 interleaved weight transform and reusable SymmBuffer allocated outside timing',
            architecture='Original SM100 BF16 algorithm with exact SM110 host-guard acceptance; real device/JIT architecture unchanged',
            mma_type='bf16xbf16', activation='swiglu', activation_clamp=None, fast_math=True,
            score_application='FP32 SwiGLU*route_score then BF16 hidden round BEFORE Down; differs from score-after-Down grain-gemm semantics',
            cumulative_stats='Initialized once, reused cumulatively as in the FP8 comparison; no per-call reset',
            communication='One rank; ordinary torch allocation, no NCCL/rendezvous/allreduce',
            num_max_tokens_per_rank=self.buffer.num_max_tokens_per_rank,
            reusable_workspace_bytes=self.buffer.buffer.numel(), output_dtype='BF16')

    def launch_full(self):
        c = self.c
        if self.buffer is None:
            raise RuntimeError('BF16 Mega experiment is closed')
        with c.torch.cuda.device(c.x.device):
            self.buffer.x[:c.tokens].copy_(c.x)
            self.buffer.topk_idx[:c.tokens].copy_(c.ids)
            self.buffer.topk_weights[:c.tokens].copy_(c.scores)
            output = c.torch.empty((c.tokens,c.hidden),device=c.x.device,dtype=c.torch.bfloat16)
            self.dg.bf16_mega_moe(y=output,l1_weights=self.w1,l2_weights=self.w2,
                sym_buffer=self.buffer,cumulative_local_expert_recv_stats=self.stats,
                activation='swiglu',activation_clamp=None,fast_math=True)
            return output

    def close(self):
        """Call after root synchronizes; views retain storage until all are dropped."""
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
        self.w1 = self.w2 = self.stats = None


def make_experiment(context):
    return Experiment(context)
