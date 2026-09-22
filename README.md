# GrainGEMM

INT8 activation × INT8 weight (W8A8) GEMM with scales grouped along the reduction dimension, K.

Python package: `grain_gemm`.

## Planned scope

For `A[M, K] @ B[K, N]`:

- Activations: one scale per row and K group.
- Weights: one scale per column and K group.
- Group sizes: **32, 64, 128, and 256**.
- Recommended balance of accuracy and inference latency on current targets: **G256**. See [Group-size recommendation](#group-size-recommendation).
- INT32 accumulation within each group, followed by scaling and FP32 accumulation across groups.
- Target GPUs: **NVIDIA A100, H100, and Jetson Thor (T5000)**.

For group size `G`, the scale shapes are `[M, ceil(K/G)]` for A and `[ceil(K/G), N]` for B. The intended operation is:

```math
C_{mn}
= \sum_{g=0}^{\lceil K/G \rceil - 1}
s^A_{mg}\,s^B_{gn}\,
\mathrm{FP32}\!\left(
\underbrace{
\sum_{k=gG}^{\min((g+1)G,\,K)-1}
A^{\mathrm{INT8}}_{mk}\,B^{\mathrm{INT8}}_{kn}
}_{\text{INT32 accumulation}}
\right)
```

Here, $s^A_{mg}$ and $s^B_{gn}$ correspond to `scale_A[m, g]` and `scale_B[g, n]`. The outer weighted sum accumulates in FP32.

Each group covers up to `G` consecutive elements along K. GrainGEMM uses a configurable block scaling scheme and does not currently claim compatibility with the standard MXINT8 format.

## Research basis

GrainGEMM's INT8 design is motivated by **[INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](https://arxiv.org/abs/2510.25602)**, Mengzhao Chen et al. (2025). See also the [authors' implementation](https://github.com/ChenMnZ/INT_vs_FP).

- **Theory ([Section 4](https://arxiv.org/html/2510.25602v1#S4)):** Smaller blocks typically reduce the peak-to-RMS ratio (crest factor), improving uniform INT8 quantization under the paper's approximate quantization signal-to-noise ratio (QSNR) model.
- **8-bit evidence ([Section 5.2](https://arxiv.org/html/2510.25602v1#S5.SS2)):** With 32-element blocks and UE8M0 scales, MXINT8 achieves lower KL divergence from BF16 than MXFP8 on all 12 evaluated models.
- **Groups 32/64/128/256 ([Table 10](https://arxiv.org/html/2510.25602v1#S11.T10)):** These sizes appear in INT8 training ablations of scale precision and symmetric `[-127, 127]` clipping. This table compares INT8 quantization recipes, without an FP8 baseline.
- **Hardware motivation ([Section 6](https://arxiv.org/html/2510.25602v1#S6)):** Gate-level modeling estimates lower MXINT8 energy and area than MXFP8 at matched throughput.

These results motivate GrainGEMM's custom K-group scaling. The practical recommendation below also accounts for the cost of applying group scales on existing GPUs.

## Group-size recommendation

**Based on project experiments, G256 (`G = 256`) offers a good balance between quantization accuracy and inference latency on A100, H100, and Jetson Thor (T5000), and is our recommended starting point for these GPUs.**

These GPUs support INT8 Tensor Core matrix multiplication, but do not expose a native MXINT8 instruction that combines INT8 dot products with per-group scaling. Group scales therefore have to be applied in software. NVIDIA's [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) documents the available integer and block-scaled MMA variants.

The software scaling path maintains two accumulator states:

- **INT32 partial sums** for the current K-group, produced by INT8 Tensor Core operations.
- **FP32 running sums** that accumulate the scaled contributions from all K-groups.

In a register-based implementation, these occupy two sets of accumulator registers, increasing register pressure. At each group boundary, the INT32 partial sums are converted to FP32, scaled by `scale_A[m, g] * scale_B[g, n]`, and added to the FP32 running sums using CUDA-core arithmetic. The INT32 accumulator is then reused for the next group. The storage depends on the backend: a Thor `tcgen05` path can keep MMA accumulators in [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma) and move fragments into registers for software scaling.

For a fixed K, this conversion-and-scaling work occurs `ceil(K / G)` times. **G256 reduces scale loads, conversions, and CUDA-core updates by performing more INT8 Tensor Core work between scaling steps.** It does not eliminate the separate accumulator states. Smaller groups can improve quantization accuracy, but require more frequent software scaling on these GPUs; G256 balances these competing costs in our experiments.

**On hardware with native MXINT8 support, prefer finer quantization groups supported by the instruction, such as G32.** Applying block scales within the matrix-multiply operation reduces the software overhead that motivates G256 on the current targets. The final choice should reflect the hardware's supported granularity, matrix shapes, and the model's accuracy and latency requirements.

## Development status

The repository includes a standalone SGLang-derived Triton baseline, correctness tests, and a benchmark entry point. GrainGEMM-specific optimized kernels and performance results for A100, H100, and Thor are still pending. The G256 recommendation above records project experimental observations and is not a measured comparison against this baseline.

## SGLang Triton baseline

[`sglang_int8_gemm`](src/grain_gemm/baselines/sglang.py) extracts SGLang's INT8 kernel at [commit `8ac19cc`](https://github.com/sgl-project/sglang/blob/8ac19cc19f8ade51ed203478f17c9a09c706d730/python/sglang/kernels/ops/quantization/int8_kernel.py). It runs independently of the SGLang package. The original kernel computation is retained; GrainGEMM supplies the input validation and launch wrapper. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and the upstream Apache-2.0 license.

- Inputs: INT8 `A[M, K]` and `B[K, N]`, with FP32 scales `[M, ceil(K/G)]` and `[ceil(K/G), N]` on the same CUDA device.
- Groups: **32, 64, 128, 256**; the default is **256**. The final group may be shorter than G.
- Accumulation: one INT32 dot product per complete K-group (or masked tail), followed by FP32 scaling and accumulation. Output may be FP32 (default), FP16, or BF16.
- Layout: transposed positive-stride views are accepted without an implicit copy. For example, weights stored as `[N, K]` can be passed as `weight.T`.

The wrapper sets `group_n=1` and `BLOCK_SIZE_K=G`. Its M/N compute tiles are independent of the quantization groups, avoiding a one-column compute tile when weights have per-column scales. The default launch configuration is a starting baseline, not an autotuned SGLang result. `block_m`, `block_n`, `num_warps`, and `num_stages` can be overridden for experiments.

Use a CUDA-enabled PyTorch build and a compatible Triton version on Linux. Triton's CUDA driver build also needs a C compiler and matching Python development headers.

```bash
python -m pip install -e ".[baseline,test]"
```

Example using prequantized inputs and their dequantization scales:

```python
import torch
from grain_gemm.baselines import sglang_int8_gemm

M, N, K, G = 128, 1024, 4096, 256
a = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
b = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8).T
scale_a = torch.full((M, K // G), 1 / 127, device="cuda")
scale_b = torch.full((K // G, N), 1 / 127, device="cuda")
c = sglang_int8_gemm(a, b, scale_a, scale_b, group_size=G)
```

Run correctness checks and compare group sizes:

```bash
python -m pytest -q
python benchmarks/bench_sglang.py --m 128 --n 4096 --k 4096 --group-size 32 64 128 256
```

The benchmark checks up to 32 evenly spaced output rows and columns against independent CPU INT32 dot products before timing. It reports median CUDA-graph GEMM latency with repeated inputs and no cache flush, excluding compilation, warmup, and input preparation. It does not measure activation quantization or end-to-end inference latency. The group-size sweep uses synthetic inputs and is not a model-accuracy evaluation. Use `--output results.json` to save the GPU, software versions, launch configuration, checks, and measurements.

Initial validation on **GB10**, with PyTorch 2.14.0+cu130 and Triton 3.8.0: **26 tests passed**. An [example benchmark report](benchmarks/results/gb10_m128_n4096_k4096_fp32.json) records all four group sizes at `M=128, N=4096, K=4096`. G64 was fastest in this single run with the default untuned configuration; this does not establish a general group-size ranking or performance on the target GPUs.

To compare the baseline with PyTorch BF16 GEMM:

```bash
python benchmarks/compare_bf16.py --m 1024 --n 2048 --k 4096 --group-size 256 --output comparison.json
```

Both paths use the same original BF16 inputs, row-major A, column-major B, and BF16 output. The INT8 path quantizes the inputs before timing. Five rounds alternate measurement order and report median CUDA-graph latency. On GB10, the default untuned INT8 configuration measured **241.14 µs**, versus **207.31 µs** for `torch.mm` in BF16: **16.3% higher latency**. See the [full report](benchmarks/results/gb10_m1024_n2048_k4096_g256_vs_bf16.json) for individual rounds, software versions, correctness checks, and synthetic-input quantization error. This comparison excludes quantization cost and does not measure end-to-end inference.

## Local development

```bash
python -m pip install -e .
python -c "import grain_gemm; print(grain_gemm.__version__)"
```

## Reference

```bibtex
@article{int_vs_fp_2025,
  title={INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats},
  author={Chen, Mengzhao and Wu, Meng and Jin, Hui and Yuan, Zhihang and Liu, Jing and Zhang, Chaoyi and Li, Yunshui and Huang, Jie and Ma, Jin and Xue, Zeyue and Liu, Zhiheng and Bin, Xingyan and Luo, Ping},
  journal={arXiv preprint arXiv:2510.25602},
  year={2025},
  url={https://arxiv.org/abs/2510.25602}
}
```
