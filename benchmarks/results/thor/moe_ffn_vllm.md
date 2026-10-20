# Thor BF16 MoE FFN: vLLM expert core used by vLLM-Omni

GPU frequency was **reduced to 1.2 GHz**, using the supported locked setting
of **1197 MHz**. The upstream vLLM BF16 Triton functional expert FFN reached
**47.19 effective TFLOPS** at 4096 tokens, taking **3.277 ms**. This measures
the expert computation used by Omni, without an Omni serving engine or model
router. No INT8 method is timed in this comparison.

NVIDIA Thor SM110, 20 SMs; E=32, H=1536, I=1024, top-k=4, ordinary SwiGLU.
Both methods receive the same BF16 weight/input values and the same numeric
expert IDs and routing scores. The Cosmos compiled comparison was remeasured
in this run; ratios do not combine latencies from earlier runs.

| Tokens | vLLM BF16 ms | vLLM effective TFLOPS | Cosmos compiled BF16 ms | Cosmos effective TFLOPS | Cosmos / vLLM |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.930 | 2.50 | 2.349 | 2.06 | 1.22x |
| 512 | 2.086 | 9.27 | 2.634 | 7.34 | 1.26x |
| 2048 | 2.620 | 29.51 | 3.247 | 23.81 | 1.24x |
| 3072 | 2.620 | 44.27 | 3.200 | 36.24 | 1.22x |
| 4096 | 3.277 | 47.19 | 3.871 | 39.94 | 1.18x |

The ratio is Cosmos latency divided by vLLM latency. This run's vLLM expert
core was 1.18x–1.26x faster across these five shapes. These are effective
**full expert FFN** throughputs, including host dispatch gaps, rather than
isolated GEMM or hardware peak TFLOPS.

## Original implementation and loading

Omni revision [`da572cddb38383deae9caef33b8d490e292ef071`](https://github.com/vllm-project/vllm-omni/tree/da572cddb38383deae9caef33b8d490e292ef071)
uses the upstream vLLM `FusedMoEFactory` through its
[diffusion adapter](https://github.com/vllm-project/vllm-omni/blob/da572cddb38383deae9caef33b8d490e292ef071/vllm_omni/diffusion/layers/fused_moe.py#L52).
The benchmark loads vLLM revision
[`4e304b4d1ed0037514073305d839a79586d0e9de`](https://github.com/vllm-project/vllm/tree/4e304b4d1ed0037514073305d839a79586d0e9de),
executing its original
[`fused_experts_impl`](https://github.com/vllm-project/vllm/blob/4e304b4d1ed0037514073305d839a79586d0e9de/vllm/model_executor/layers/fused_moe/fused_moe.py#L1652):
expert alignment/counting, two Triton expert GEMMs, CUDA SiLU/multiply and CUDA
FP32 top-k reduction. The original GPU source bodies and tile-selection
functions are unchanged. The selected source files are clean in both checkouts.

The full vLLM package is not installed in this environment. The optional loader
executes selected original AST definitions with their original filenames and
line numbers. It supplies isolated CUDA platform queries, a default tile-override
context, and a private activation namespace. The native helper builds the
unchanged `moe_align_sum_kernels.cu` and `activation_kernels.cu` plus original
local headers, adding only three operator registrations in a separate namespace.
No sorting, activation, GEMM or reduction algorithm is replaced with a reference.
Native build products stay outside the repository.

The standalone runner validates inputs and directly calls `fused_experts_impl`,
bypassing vLLM's serving custom-op wrapper. Its workspaces are allocated per
call; a full modular MoE runner can reuse workspaces. Consequently these
measurements do not establish full Omni model latency or the performance of
an engine using CUDA Graphs. MAGI-2's head-local SwiGLU7 implementation is a
different operation and is not used here.

The default CUDA tensor-descriptor option is disabled. The original lookup found
no `E=32,N=1024,device_name=NVIDIA_Thor.json`, so all five cases use the upstream
`get_default_config`, with no custom tuning or overridden tile:

| Tokens | Block M | Block N | Block K | Warps | Stages | Group M |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 128 | 64 | 4 | 3 | 1 |
| 512 | 64 | 128 | 64 | 8 | 3 | 1 |
| 2048, 3072, 4096 | 128 | 128 | 64 | 8 | 3 | 1 |

Cosmos uses the same pinned original grouped-MM class, histogram and padding
sources described in the [earlier Cosmos comparison](moe_ffn_cosmos.md), at
revision `4ed6244506e00bb270ed17d4532897f2aaa7bc16`. It uses
`torch.compile(fullgraph=True, dynamic=True)` in default Inductor mode, without
additional autotuning or CUDA Graph capture.

## Timing, layouts and checks

Every timed call includes expert dispatch/alignment, gate/up, activation, down
and combination. Router logits/top-k selection, weight preparation, compilation,
warmup, numerical checks and cooling are outside timing. Each method receives
at least 200 ms of untimed warmup. Five rounds rotate method order; each round
takes the median of five full-call CUDA-event samples, and the table uses the
median of those five round medians. JSON retains all 25 samples per method
and shape, execution order and per-round GPU states.

Effective TFLOPS is `6*T*top_k*H*I / (latency_seconds*1e12)`: gate/up contributes
`4*T*top_k*H*I` and down contributes `2*T*top_k*H*I`. The count uses valid routes,
excludes extra padding and does not count activation/routing operations. Repeated
operands are used without an explicit cache flush.

vLLM uses original Linear weights `[E,2I,H]` and `[E,H,I]`, with their last
dimension contiguous. Cosmos retains its upstream N-contiguous transposed
weight storage; conversion happens once before timing. Scores are normalized
once in FP32 and rounded to BF16 before timing: Cosmos receives BF16, while
vLLM receives the identical values in FP32. Cosmos multiplies scores before
down; vLLM multiplies the FP32 down accumulator before converting it to BF16.
vLLM also rounds SiLU to BF16 before multiplying up, and sums route outputs
in FP32. These different rounding points preclude bitwise agreement.

Every shape passed independent CPU sampling of two tokens/eight routes.
vLLM uses CPU FP64 full-H/I dot products with explicit BF16 rounding points;
Cosmos compiled uses the existing CPU reference with fused-pointwise rounding.
Checks use `rtol=0.025`, `atol=0.008`. Sampled relative L2 error was at most
**0.083%** for vLLM and **0.322%** for Cosmos. All full outputs were finite.
The full-output vLLM/Cosmos relative L2 difference was approximately **0.48%**;
this synthetic-input check does not measure model quality.

Three additional tests passed: original Python/Triton source identity and
native provenance, plus complete FFN comparisons with INT32 and INT64 routing
indices. They cover duplicate expert selections, empty experts, nonnormalized
FP32 scores and an all-zero-score token. Recorded temperatures across
before/after and per-round states ranged from **49.66 to 60.12 C**; clocks
remained **1197 MHz**, and no thermal slowdown was reported.

Software: PyTorch `2.11.0a0+a6c236b9fd.nv26.03.46836102`, CUDA 13.2,
Triton 3.6.0, Python 3.12.3. Original CUDA sources were built with NVCC 13.2,
SM110a, C++20, O3/NDEBUG and the upstream-compatible Torch stable-ABI/half
flags, without `--use_fast_math`. All flags, source/build/library hashes and
dependency-isolation choices are recorded in JSON. No production GrainGEMM
kernel, GB10 tuning or earlier performance data was changed.

## Reproduction

Use the three pinned external checkouts above and a CUDA-enabled PyTorch build
with its headers plus NVCC. Configure the host GPU clock to 1197 MHz before
running; the script records clocks and does not change them.

```bash
PYTHONPATH=src python benchmarks/bench_moe_vllm.py \
  --vllm-dir /path/to/vllm --omni-dir /path/to/vllm-omni \
  --cosmos-dir /path/to/cosmos-framework \
  --tokens 128 512 2048 3072 4096 \
  --output benchmarks/results/thor/moe_ffn_vllm.json

GRAIN_VLLM_DIR=/path/to/vllm PYTHONPATH=src \
  python -m pytest -q tests/test_vllm_moe_baseline.py
```

[Raw results, samples, source and build provenance](moe_ffn_vllm.json) ·
[CSV](moe_ffn_vllm.csv) · [Standalone BF16 GEMM](bf16_gemm_1197mhz.md).
