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

```text
C[m, n] = sum_g (
    scale_A[m, g] * scale_B[g, n]
    * float32(sum_{k in group g} int8_A[m, k] * int8_B[k, n])
)
```

Each group covers up to `G` consecutive elements along K. These group sizes are a custom block scaling scheme; they do not claim compatibility with the standard MXINT8 format.

## Development status

This repository currently contains an initial project scaffold. GPU kernels are not implemented, hardware support is not validated, and no performance results are available.

## Local development

```bash
python -m pip install -e .
python -c "import grain_gemm; print(grain_gemm.__version__)"
```
