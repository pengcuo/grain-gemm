# Thor MoE FFN: INT8 versus Cosmos Framework BF16

GPU frequency was **reduced to 1.2 GHz**; the actual supported locked setting
was **1197 MHz**. The primary floating-point comparison is the original Cosmos
Framework grouped-MM expert FFN with its default compilation options.

Across these five token counts, the compiled BF16 / INT8 latency ratio ranges
from **0.90x to 1.81x**. INT8 is faster on four shapes; at **4096 tokens,
compiled Cosmos BF16 is faster**, taking 3.873 ms versus INT8's 4.296 ms.
This comparison replaces the per-expert-loop baseline as the relevant Cosmos
comparison; it does not establish a speedup for every MoE workload.

NVIDIA Thor SM110, E=32 experts, H=1536, I=1024, top-k=4, G=256 and SwiGLU.
The BF16 input tensors and original weight values are shared across methods.
INT8 weights and scales are prepared once outside timing.

| Tokens | INT8 auto ms | INT8 effective TOPS | Cosmos BF16 eager ms | Cosmos BF16 compiled ms | Compiled BF16 effective TFLOPS | Compiled BF16 / INT8 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.153 | 4.19 | 2.365 | 2.091 | 2.31 | 1.81x |
| 512 | 1.321 | 14.63 | 2.295 | 1.887 | 10.24 | 1.43x |
| 2048 | 2.550 | 30.31 | 3.815 | 2.701 | 28.63 | 1.06x |
| 3072 | 3.486 | 33.27 | 5.398 | 3.931 | 29.50 | 1.13x |
| 4096 | 4.296 | 35.99 | 6.001 | 3.873 | 39.92 | 0.90x |

The 3072/4096 rows were added in a separate process using the same protocol,
production kernels, benchmark sources, weights and upstream Cosmos revision.
The 3072 row additionally includes a second independent run because its
full-call latency varied substantially across rounds: it pools all ten round
medians from both five-round runs and takes their median, without discarding
either run. All other rows retain five-round medians. The two individual 3072
run medians were INT8 **3.675 / 3.303 ms**, eager BF16 **5.618 / 5.179 ms** and
compiled BF16 **3.910 / 3.952 ms**. Treat the 3072 speedup as a variable full-call
measurement; it is not a stable isolated-kernel speedup.
Across its ten rounds, INT8 round medians ranged from **3.274 to 4.133 ms**,
and compiled BF16 from **3.242 to 4.349 ms**.

The ratio is BF16 latency divided by INT8 latency; values above one favor INT8.
INT8 auto selects ragged Triton at 128/512 tokens and the Thor native batched
CUTLASS path at 2048/3072/4096 tokens (capacity 320/448/640 per expert,
respectively). The BF16 methods retain
Cosmos's variable-size groups with each expert padded independently to a
multiple of 16; they do not loop over experts in Python.

## Source and execution

The source checkout is official `NVIDIA/cosmos-framework` revision
[`4ed6244506e00bb270ed17d4532897f2aaa7bc16`](https://github.com/NVIDIA/cosmos-framework/tree/4ed6244506e00bb270ed17d4532897f2aaa7bc16).
The baseline executes its unchanged `Qwen3VLMoeTextExpertsGroupedMm` class,
`_run_experts_grouped_mm`, `_weighted_expert_counts` and original Triton padding
module. This preserves stable sorting, sentinel/gather dispatch, two
`torch._grouped_mm` calls, SwiGLU, score multiplication before down and BF16
scatter-add. Source references: [expert FFN](https://github.com/NVIDIA/cosmos-framework/blob/4ed6244506e00bb270ed17d4532897f2aaa7bc16/cosmos_framework/model/generator/reasoner/qwen3_vl_moe/moe.py#L40),
[expert counts](https://github.com/NVIDIA/cosmos-framework/blob/4ed6244506e00bb270ed17d4532897f2aaa7bc16/cosmos_framework/model/generator/reasoner/qwen3_vl_moe/qwen3_vl_moe.py#L322),
[compile defaults](https://github.com/NVIDIA/cosmos-framework/blob/4ed6244506e00bb270ed17d4532897f2aaa7bc16/cosmos_framework/model/generator/reasoner/qwen3_vl_moe/moe_bench.py#L266).

The benchmark's loader isolates unrelated framework imports by extracting the
original function/class AST nodes without changing their bodies or source line
numbers. The Triton module is loaded directly from the checkout. This environment
lacks Transformers; the SiLU provider executes the same `nn.functional.silu`
operation as Transformers 4.57.6's SiLUActivation. The loader choice, activation
provider, five upstream source hashes and source revision are recorded in JSON.
Cosmos retains its original N-contiguous BF16 Parameters; copies of the same
Linear weight values are made before timing. INT8 retains its public
K-contiguous transposed weight views.

`Cosmos BF16 compiled` uses `torch.compile(fullgraph=True, dynamic=True)` with
the default Inductor mode and no added max-autotune setting. The compiled
boundary includes the expert histogram and the original expert forward.
Unlike Cosmos's isolated expert microbenchmark, both eager and compiled calls
here include expert counts, because the INT8 public API computes those counts
internally. There is no silent substitution with a per-expert reference when a
Cosmos operator or compilation fails. No CUDA Graph capture is used.

## Timing and numerical checks

Every timed call includes counts, sorting/padding/gather or equivalent dispatch,
both projections, activation and combination. INT8 additionally includes dynamic
activation quantization. Both exclude router logits/top-k selection, input
creation, weight preparation, compilation, correctness checks, warmup and cooling.
These are **pre-routed full FFN** latencies including host dispatch gaps, rather
than GEMM-only throughput. INT8 is measured through its eager public auto API;
Cosmos is shown both eager and compiled.

Each method receives at least 200 ms of untimed warmup after cooling. Five rounds
per run rotate the three-method execution order. Each round takes the median of
five full-call CUDA event samples; the table reports the median of round medians
(ten round medians for the repeated 3072 case, as explained above).
Effective throughput uses `6*T*top_k*H*I / latency`, excluding non-GEMM operation
counts. This counts both gate/up (`4*T*top_k*H*I`) and down
(`2*T*top_k*H*I`), using valid routes and excluding extra padding work.
Recorded temperatures ranged from approximately 49 to 54 C, clocks
remained locked at 1197 MHz and no thermal slowdown was reported.

Scores are normalized once in FP32 and rounded to BF16 before timing. Cosmos
receives BF16 scores and INT8 receives FP32 scores with identical numeric values;
this avoids charging Cosmos an unnecessary score conversion. Cosmos applies
these scores **before down**;
INT8 applies them **after down** and combines in FP32. These contracts have the
same real-valued linear formula, but differ in quantization and rounding.
Across-method equality is therefore not asserted. INT8 versus compiled Cosmos
BF16 has a relative L2 difference of approximately **2.27%** and RMSE around
**0.0077** on these synthetic inputs. These figures do not measure model quality.

Each shape passed independent CPU sampling of two tokens/eight routes: INT8
against INT32 group dots, Cosmos eager against a BF16 projection/pointwise
reference, and Cosmos compiled against a fused-pointwise BF16 reference.
All complete outputs were finite. Five additional baseline tests passed,
covering eager/compiled execution, duplicate routes, empty experts, masks and
masked NaN inputs, plus source/stride/count verification. The production INT8
API, kernels and previous single-GEMM performance data were not modified.

## Reproduction

```bash
PYTHONPATH=src python benchmarks/bench_moe_cosmos.py \
  --cosmos-dir /path/to/cosmos-framework \
  --output benchmarks/results/thor/moe_ffn_cosmos.json

PYTHONPATH=src python benchmarks/bench_moe_cosmos.py \
  --cosmos-dir /path/to/cosmos-framework --tokens 3072 4096 \
  --output benchmarks/results/thor/moe_ffn_cosmos_large_tokens.json

PYTHONPATH=src python benchmarks/bench_moe_cosmos.py \
  --cosmos-dir /path/to/cosmos-framework --tokens 3072 \
  --output benchmarks/results/thor/moe_ffn_cosmos_3072_repeat.json

GRAIN_COSMOS_DIR=/path/to/cosmos-framework PYTHONPATH=src \
  python -m pytest -q tests/test_cosmos_moe_baseline.py
```

The benchmark records clocks and does not change them. Use the pinned upstream
revision to reproduce this source version.

- Original 128/512/2048 tokens: [raw results, rounds and provenance](moe_ffn_cosmos.json),
  [CSV](moe_ffn_cosmos.csv).
- Added 3072/4096 tokens: [raw results, rounds and provenance](moe_ffn_cosmos_large_tokens.json),
  [CSV](moe_ffn_cosmos_large_tokens.csv).
- Repeated 3072 tokens: [raw results, rounds and provenance](moe_ffn_cosmos_3072_repeat.json),
  [CSV](moe_ffn_cosmos_3072_repeat.csv).
- [Earlier per-expert-loop comparison](moe_ffn.md)
- [Public API and numerical contract](../../../docs/moe_ffn.md)
