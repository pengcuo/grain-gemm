# Thor INT8 MoE FFN measurements

For the original Cosmos Framework grouped-MM BF16 baseline, use the newer
[Cosmos comparison](moe_ffn_cosmos.md). This report retains the earlier
per-expert-loop baseline and its separate measurement run.

GPU frequency was **reduced to 1.2 GHz**; the actual supported locked setting
was **1197 MHz** throughout this run. This is an end-to-end FFN measurement,
separate from the square-GEMM protocol in this directory.

NVIDIA Thor SM110; BF16 tokens; E=32 experts, H=1536, I=1024, top-k=4,
G=256 and SwiGLU. Expert weights are prequantized INT8, with explicit FP32
scales in the public GEMM layout. Routing is ragged and generated from a fixed
seed (2026); no equal token count is imposed on the caller.

| Tokens | Auto implementation | INT8 ms | INT8 effective TOPS / TFLOPS equiv. | PyTorch eager BF16 ms | BF16 effective TFLOPS | Eager BF16 / INT8 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | ragged Triton | 0.989 | 4.89 | 4.502 | 1.07 | 4.55x |
| 512 | ragged Triton | 1.585 | 12.19 | 5.076 | 3.81 | 3.20x |
| 2048 | SM110 batched CUTLASS #32 | 2.651 | 29.16 | 6.366 | 12.14 | 2.40x |

The BF16 baseline is a readable PyTorch per-expert loop using `torch.where`,
two Linear operations, SwiGLU and FP32 `index_add_`. It is **not an optimized
fused BF16 MoE kernel**; these ratios describe only this baseline.

Both timings include routing, both projections, activation and weighted
combination; INT8 also includes dynamic input/intermediate quantization. Weight
preparation, router/top-k calculation, compilation and warmup are excluded.
Each path receives at least 200 ms of untimed warmup after cooling. Five rounds
alternate execution order; each round takes the median of five full-call CUDA
event samples, and the table reports the median of those round medians. These
are eager calls and include host dispatch gaps and the API's single routing
statistics synchronization. CUDA Graphs are not used for timing. Throughput is
`6 * T * top_k * H * I / latency`; non-GEMM operation counts are excluded.

Recorded temperatures stayed below 53 C, with no reported thermal slowdown.
The raw results retain every round median, frequency/temperature snapshots,
route counts, software versions and source/native hashes. The recorded Git
revision is the base revision before this change; the source hashes identify
the measured implementation.

Each shape passed independent CPU INT32 reference sampling of two tokens
(eight selected expert slots). Full-output RMSE against unquantized eager BF16
was approximately 0.0077; this is quantization error, separate from agreement
with the INT8 reference. The correctness suite additionally covers BF16/FP16,
duplicate routes, signed/zero scales, masked and empty experts, non-default
streams, native batch scale strides and CUDA Graph replay with changed routing.
The full regression run passed 1022 tests with 15 skips. This includes fused
routing statistics, extreme INT64 ids, masked ids, empty/multi-block routing,
native tile-metadata omission and deferred tiles on skewed-routing fallback.
A targeted Compute Sanitizer memcheck run passed 27 cases with **0 errors**.

- [Raw results and provenance](moe_ffn.json)
- [CSV table and timing rounds](moe_ffn.csv)
- [API, strides and numerical behavior](../../../docs/moe_ffn.md)

Reproduce from the repository root after building the optional SM110 CUTLASS
backend:

```bash
PYTHONPATH=src python benchmarks/bench_moe_ffn.py \
  --output benchmarks/results/thor/moe_ffn.json
```

The benchmark records frequency and does not change it. The table's auto
implementations follow the documented dispatch rule and the recorded route
counts: the largest shape uses capacity 320 per expert; the smaller shapes
use ragged Triton. Explicit native paths never silently fall back to Triton.

The native projections retain the Cosmos cfg20/21 mainloop, mapped to GrainGEMM
CUTLASS configs 32/33; the [API documentation](../../../docs/moe_ffn.md) gives the
source revision and pipeline details. Routing validation and native capacity now
share one GPU statistics read. Native paths skip the three ragged tile metadata
arrays; auto builds them only when its padding check falls back to Triton.
Quantization, projection rounding, scale layout and weighted-combination order
are unchanged. A separate same-process comparison against revision `8dba0e1`
confirmed bitwise-identical complete outputs for all three shapes. Host dispatch
variation limits claims about small latency differences; the table above is a
fresh full-call measurement against the stated eager BF16 baseline.
