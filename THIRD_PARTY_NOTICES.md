# Third-party notices

## SGLang Triton INT8 GEMM

`src/grain_gemm/baselines/sglang.py` contains the
`_w8a8_block_int8_matmul` kernel extracted from
[SGLang](https://github.com/sgl-project/sglang), commit
`8ac19cc19f8ade51ed203478f17c9a09c706d730`:

- [Original source](https://github.com/sgl-project/sglang/blob/8ac19cc19f8ade51ed203478f17c9a09c706d730/python/sglang/kernels/ops/quantization/int8_kernel.py)
- Copyright 2023-2024 SGLang Team, as stated in the upstream repository license.
- Licensed under the Apache License, Version 2.0. The complete upstream
  license is included in [licenses/SGLang-LICENSE](licenses/SGLang-LICENSE).

GrainGEMM preserves the extracted kernel's computation and provides a new
standalone wrapper. The wrapper uses logical `A[M, K] @ B[K, N]` layouts,
validates inputs, handles empty matrices, and launches with `group_n=1`
and `BLOCK_SIZE_K=group_size`. Compute tiles along M and N are independent
of the quantization group size. SGLang imports, configuration lookup, and
unrelated quantization kernels are omitted. These are GrainGEMM adaptations,
not an upstream SGLang performance configuration or endorsement.

## CUTLASS CuTe CUDA implementation

`src/grain_gemm/kernels/csrc/grain_cute.cu` adapts the
[CUTLASS SM80 CuTe GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/sgemm_sm80.cu),
commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
The upstream copyright and BSD-3-Clause notice are retained in the source;
the full license is included in [licenses/CUTLASS-LICENSE](licenses/CUTLASS-LICENSE).

GrainGEMM changes the operands to signed INT8, accumulates each 256-element
K-group in INT32, applies row/column scales into an FP32 running accumulator,
and writes BF16 output. It adds launch configurations, tile scheduling, and a
C ABI launcher. CUTLASS headers are an external build dependency; they are not
vendored into this repository. The SGLang baseline remains a separate reference
implementation.
