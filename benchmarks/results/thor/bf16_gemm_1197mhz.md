# Standalone Thor BF16 GEMM at 1.2 GHz

GPU compute frequency was **reduced to 1.2 GHz**, using the supported locked
setting of **1197 MHz**. On this NVIDIA Thor SM110 (20 SMs), the highest
throughput in this run was **130.32 TFLOPS** for a 2048-square BF16 GEMM with
N-contiguous B and a preallocated output. The corresponding ordinary
`torch.mm(A, weight.T)` call reached **129.25 TFLOPS**.

These are fresh measurements on 2026-10-20, separate from the earlier INT8/BF16
comparisons. They measure standalone GEMM execution, rather than full MoE FFN
latency or a theoretical hardware peak.

| M = N = K | NT `torch.mm` TFLOPS | NT `out=C` TFLOPS | NN `torch.mm` TFLOPS | NN `out=C` TFLOPS | Best median latency, us |
|---:|---:|---:|---:|---:|---:|
| 1024 | 123.67 | 122.47 | 123.58 | 123.01 | 17.36 |
| 2048 | 129.25 | 129.18 | 129.57 | 130.32 | 131.83 |
| 4096 | 128.08 | 124.07 | 128.59 | 127.25 | 1068.85 |
| 8192 | 116.94 | 117.19 | 117.34 | 117.55 | 9353.86 |

A has logical shape `[M,K]`, row-major stride `(K,1)`. B has logical shape
`[K,N]`: **NT** uses a transposed contiguous Linear weight, `weight.T`, with
stride `(1,K)` (K contiguous, matching GrainGEMM's public weight layout);
**NN** uses a contiguous copy with stride `(N,1)` (N contiguous). Both layouts
contain the same numeric values. All inputs and outputs are BF16; `out=C`
uses a preallocated BF16 output. Layout preparation and output preallocation
are outside timing.

Each method receives at least 200 ms of untimed warmup per shape. Three rounds
rotate method order; every table entry uses the median of round medians from
`triton.testing.do_bench_cudagraph`. The requested captured-batch duration is
10 ms, and the installed helper replays each batch ten times. CUDA Graph
timing measures GPU GEMM execution and excludes host dispatch, input creation,
layout preparation, validation and warmup. Repeated inputs and weights are
used without an explicit L2 flush. Throughput is `2*M*N*K / (seconds*1e12)`;
one multiply-add counts as two floating-point operations. The reported maximum
selects among these per-method medians, rather than selecting a single fastest
sample.

Software: PyTorch `2.11.0a0+a6c236b9fd.nv26.03.46836102`, CUDA runtime 13.2,
Triton 3.6.0, Python 3.12.3. The recorded PyTorch BLAS preference is cuBLAS;
BF16 reduced-precision reduction is enabled (the existing setting). The
benchmark does not change precision policy, clocks or power policy.

Before timing, every method passed an independent CPU FP64 dot-product check
of 64 output elements, using the full K dimension and a BF16-rounded reference
(`rtol=0.016`, `atol=0.005`). All complete outputs were finite. Recorded sample
relative L2 differences were at most 0.000219. GPU clock snapshots remained at
1197 MHz. Before/after shape temperatures ranged from 49.62 to 72.06 C, and
thermal slowdown was not active in the checked states. Cooling to at most
60 C before each shape was outside timing. This run did not include 16384.

The earlier [Cosmos MoE comparison](moe_ffn_cosmos.md) reports **effective full
FFN throughput**: its denominator also includes dispatch, activation, combination
and host gaps (plus activation quantization for INT8). Its lower TFLOPS values
therefore describe a different workload and timing scope. Square GEMM results
do not establish the achievable throughput of small per-expert grouped GEMMs.

Reproduce from the repository root, after configuring the host GPU clock to
1197 MHz:

```bash
python benchmarks/bench_bf16_gemm.py \
  --sizes 1024 2048 4096 8192 --rounds 3 --rep-ms 10 --warmup-ms 200 \
  --output benchmarks/results/thor/bf16_gemm_1197mhz.json
```

[Raw results, timing rounds, strides and source hashes](bf16_gemm_1197mhz.json)
and [CSV](bf16_gemm_1197mhz.csv).
