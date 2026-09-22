# GrainGEMM

INT8 activation × INT8 weight (W8A8) GEMM with scales grouped along the reduction dimension, K.

Python package: `grain_gemm`.

## Planned scope

For `A[M, K] @ B[K, N]`:

- Activations: one scale per row and K group.
- Weights: one scale per column and K group.
- Group sizes: **64, 128, and 256**.
- INT32 accumulation within each group, followed by scaling and FP32 accumulation across groups.
- Target GPUs: **NVIDIA A100, H100, and Thor**.

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

Each group covers up to `G` consecutive elements along K. These group sizes are a custom block scaling scheme; they do not claim compatibility with the standard MXINT8 format.

## Research basis

GrainGEMM's INT8 design is motivated by **[INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](https://arxiv.org/abs/2510.25602)**, Mengzhao Chen et al. (2025). See also the [authors' implementation](https://github.com/ChenMnZ/INT_vs_FP).

- **Theory ([Section 4](https://arxiv.org/html/2510.25602v1#S4)):** Smaller blocks typically reduce the peak-to-RMS ratio (crest factor), improving uniform INT8 quantization under the paper's approximate quantization signal-to-noise ratio (QSNR) model.
- **8-bit evidence ([Section 5.2](https://arxiv.org/html/2510.25602v1#S5.SS2)):** With 32-element blocks and UE8M0 scales, MXINT8 achieves lower KL divergence from BF16 than MXFP8 on all 12 evaluated models.
- **Groups 64/128/256 ([Table 10](https://arxiv.org/html/2510.25602v1#S11.T10)):** These sizes appear in INT8 training ablations of scale precision and symmetric `[-127, 127]` clipping. This table compares INT8 quantization recipes, without an FP8 baseline.
- **Hardware motivation ([Section 6](https://arxiv.org/html/2510.25602v1#S6)):** Gate-level modeling estimates lower MXINT8 energy and area than MXFP8 at matched throughput.

These results motivate GrainGEMM's custom K-group scaling. Its scale choices, numerical behavior, and kernel performance on A100/H100/Thor still require validation.

## Development status

This repository currently contains an initial project scaffold. GPU kernels are not implemented, hardware support is not validated, and no performance results are available.

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
