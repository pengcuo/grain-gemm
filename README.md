# GrainGEMM

INT8 activation × INT8 weight (W8A8) GEMM with scales grouped along the reduction dimension, K.

Python package: `grain_gemm`.

## Operation

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

GrainGEMM includes a custom Triton kernel and an optional native CUDA/CuTe kernel, with a measured G256 dispatch table for **GB10 (SM121)**. The public API selects a backend and launch configuration by architecture, shape, dtype, and layout. A100, H100, and Thor use the portable Triton path pending validation and dedicated tuning on those GPUs.

## GrainGEMM API

Run from the repository root with CUDA-enabled PyTorch and an SM80-or-newer
NVIDIA GPU:

```bash
python -m pip install -e ".[runtime]"
```

This complete example starts with BF16 activations `x[M, K]` and a linear-layer
weight `weight[N, K]`. Replace the random tensors with your own activations and
weights. The helper quantizes each row independently in groups along K; it is
example code, not part of the `grain_gemm` API.

```python
import torch
from grain_gemm import int8_gemm


def quantize_k_groups(x, group_size):
    # This simple helper requires complete K groups.
    assert x.ndim == 2 and x.shape[1] % group_size == 0
    blocks = x.float().reshape(x.shape[0], -1, group_size)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-12) / 127
    quantized = (blocks / scales[..., None]).round().clamp(-127, 127)
    return quantized.to(torch.int8).reshape(x.shape).contiguous(), scales.contiguous()


M, N, K, G = 1024, 1024, 1024, 256
torch.manual_seed(2026)
x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

a, scale_a = quantize_k_groups(x, G)                  # [M, K], [M, K/G]
weight_int8, weight_scales = quantize_k_groups(weight, G)
b = weight_int8.T                                    # [K, N], column-major view
scale_b = weight_scales.T.contiguous()                # [K/G, N], FP32

c = int8_gemm(
    a, b, scale_a, scale_b, group_size=G,
    output_dtype=torch.bfloat16,
)
print(c.shape, c.dtype)  # torch.Size([1024, 1024]) torch.bfloat16
```

`int8_gemm` returns BF16 by default; `output_dtype=torch.bfloat16` above is
optional. Each K-group accumulates in INT32, and the scaled group results
accumulate in FP32 before conversion to the output dtype. Set
`output_dtype=torch.float32` or `output_dtype=torch.float16` explicitly for
FP32 or FP16 output; these output types use the Triton path with `backend="auto"`.

For each group, the helper uses symmetric INT8 quantization with
`scale = max(abs(values)) / 127`, a positive floor for zero groups, and
`q = round(values / scale)` clipped to `[-127, 127]`. The FP32 scales restore
the approximate original values through `values ≈ q * scale`.

| Tensor | Dtype | Shape in this example | Meaning |
| --- | --- | --- | --- |
| `a` | INT8 | `[M, K]` | Quantized activations from `x` |
| `b` | INT8 | `[K, N]` | Transposed quantized weights; each column corresponds to one row of `weight` |
| `scale_a` | FP32 | `[M, K/G]` | One activation scale per row and K group |
| `scale_b` | FP32 | `[K/G, N]` | One weight scale per K group and column of `b` |
| `c` | BF16 | `[M, N]` | Output allocated and returned by `int8_gemm`, approximating `x @ weight.T` |

`c` includes quantization error relative to multiplying the original BF16 inputs.
Keeping `b = weight_int8.T` preserves the column-major layout used by the native
backend; the transposed weight scales are made contiguous separately. The helper
above requires `K % G == 0`; the Triton GEMM implementation also supports partial
final groups with scale shapes `[M, ceil(K/G)]` and `[ceil(K/G), N]`.

`backend="auto"` uses the measured GB10 configuration where applicable. The
optional CuTe backend requires a local CUDA build; otherwise the API uses Triton.
`backend="triton"` or `backend="cuda"` explicitly selects an implementation.
`backend="cutlass"` selects the separately built CUTLASS collective implementation;
see [build and comparison instructions](benchmarks/cutlass.md).
Group sizes 32/64/128/256, positive-stride views, K tails, and FP32/FP16/BF16
outputs are supported through the Triton path. The native fast path currently
supports G256 and BF16 output with aligned row-major A, column-major B,
contiguous scales, M/N multiples of 64, and K a multiple of 256.

See [Kernel design and native build](docs/kernel_design.md) for the implementation,
CUTLASS build instructions, and dispatch behavior. Inputs must already be
quantized; the API does not perform quantization or provide autograd.

See [Tests and benchmarks](benchmarks/README.md) for validation, performance results, plots, and reproduction instructions.

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
