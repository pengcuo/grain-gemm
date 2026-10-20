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

`src/grain_gemm/kernels/csrc/sm12x/int8_g256_cute.cu` adapts the
[CUTLASS SM80 CuTe GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/sgemm_sm80.cu),
commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
The upstream copyright and BSD-3-Clause notice are retained in the source;
the full license is included in [licenses/CUTLASS-LICENSE](licenses/CUTLASS-LICENSE).

GrainGEMM changes the operands to signed INT8, accumulates each 256-element
K-group in INT32, applies row/column scales into an FP32 running accumulator,
and writes BF16 output. It adds launch configurations, tile scheduling, and a
C ABI launcher. The CUTLASS checkout is an external build dependency. The
Thor collective's three shadow headers are described below. The SGLang baseline remains a separate reference
implementation.

## CUTLASS collective CUDA implementation

`src/grain_gemm/kernels/csrc/sm12x/cutlass/collective.hpp` adapts
[CUTLASS's SM80 multistage collective](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/collective/sm80_mma_multistage.hpp).
`sm12x/cutlass/kernel.hpp` extends the device composition in
[the SM70/SM80 GemmUniversal kernel](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/kernel/sm70_gemm.hpp).
Both use commit `098de2a652cf8f00fd70b2df54051c7eccbb855a` and retain the
upstream BSD-3-Clause notice. The full license is included in
[licenses/CUTLASS-LICENSE](licenses/CUTLASS-LICENSE).

GrainGEMM adds INT32 K256 group reductions, FP32 row/column scaling and running
sums, and a bridge that passes scale parameters and FP32 fragments to the
collective, with optional asynchronous scale staging. `sm12x/cutlass/int8_g256.cu`
composes this collective with `GemmUniversalAdapter` and either CUTLASS's
vectorized epilogue or the BF16 shared/direct variants in
`sm12x/cutlass/epilogue.hpp`. It is independent of `sm12x/int8_g256_cute.cu`.

The source snapshots and ablation variants under `experiments/cutlass/` retain
the corresponding source notices and the same CUTLASS dependency. Archived
GrainGEMM package snapshots may also contain the SGLang baseline described
above. These archives preserve measurement provenance; they do not include
the external CUTLASS checkout or precompiled libraries.

## Thor CuTe Tensor Memory implementation

`src/grain_gemm/kernels/csrc/sm110/int8_g256_cute.cu`,
`sm110/int8_g256_pipeline.cuh`, and `sm110/int8_g256_pipeline_1sm.cuh` adapt
CuTe primitives from CUTLASS's Blackwell tutorials at the same pinned commit:

- [01_mma_sm100.cu](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/blackwell/01_mma_sm100.cu)
- [02_mma_tma_sm100.cu](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu)
- [04_mma_tma_2sm_sm100.cu](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu)
- [05_mma_tma_epi_sm100.cu](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu)

The upstream BSD-3-Clause copyright and license notices are retained in each
source. The complete license is included in
[licenses/CUTLASS-LICENSE](licenses/CUTLASS-LICENSE).
GrainGEMM changes the operands to signed INT8, targets SM110a, adds a bounded
asynchronous group pipeline, per-group FP32 scaling, coalesced BF16 output and
configuration dispatch. The new Thor-only CuTe IDs 26–29 implement independent
one-SM and two-SM pipelines with four producer and eight consumer warps per
CTA, two TMEM buffers, three prefetched register fragments, direct public
FP32 scale layouts and explicit fused multiply-add promotion. The two-SM
pipeline supports a final N128 half-tile through zero-filled out-of-range
operands/scales and masked output stores while preserving barrier participation;
one-SM candidates require full tiles. Their pipeline
design also draws on the Thor Cosmos G256 implementation at commit
`8977529aa23f70f02bfc36559599f00d59792996`, described below. They use CuTe
primitives directly and do not invoke the separate CUTLASS collective or GEMM
adapter. The launcher also retains the existing warp-MMA candidates described
above. CUTLASS headers remain an external build dependency; SM120/GB10 source
and configurations are unchanged.

## Thor CUTLASS blockwise implementation

`src/grain_gemm/kernels/csrc/sm110/int8_g256_cutlass.cu` and the three
shadow headers under `sm110/include/cutlass/` adapt the Thor blockwise
implementation in `nvidia-cosmos/cosmos-int8-gemm`, commit
`8977529aa23f70f02bfc36559599f00d59792996`, from
`docs/quantization/tools/cutlass_int8_sm110`. The source headers retain
NVIDIA's full BSD-3-Clause copyright and license notice. The complete license
is also included in [licenses/CUTLASS-LICENSE](licenses/CUTLASS-LICENSE).
The Cosmos patches derive from CUTLASS 4.8.0 at commit `147295a3`; this
integration is built against the GrainGEMM CUTLASS commit pinned above.

The shadow headers adapt CUTLASS's
`gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp`,
`gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp`, and
`arch/reg_reconfig.h`. They add FP32 scale factors and promoted accumulators
for INT8 MMA, pipelined Tensor Memory loads, two promotion warpgroups,
vectorized column-scale loads, and SM110 register reallocation. GrainGEMM
uses the G256 per-column configurations, accepts its existing K-major
activation-scale and MN-major weight-scale layouts directly, and makes the
normal promotion's FP32 fused multiply-add explicit. Experimental bias,
separable scale, and reduced-precision scale schemes are disabled. The C ABI
accepts caller-owned workspace on each launch, preserving CUDA Graph and
stream ownership, and retains the original 32 warp-MMA configurations.

`src/grain_gemm/kernels/csrc/sm110/int8_g256_moe.cu` composes the same
Cosmos-derived G256 collective into a separate MoE library. It retains the
upstream BSD-3-Clause notice and uses smaller one-SM/two-SM tiles and persistent
schedulers for expert matrices. The public weight and scale layouts, INT32
group dots and FP32 scaled accumulation are preserved. Its configurations do
not replace the single-GEMM configurations or the SM12x implementations.

## Optional upstream vLLM BF16 MoE benchmark

`benchmarks/vllm_moe_baseline.py` loads unchanged function definitions and
Triton kernels from an external [vLLM](https://github.com/vllm-project/vllm)
checkout at commit `4e304b4d1ed0037514073305d839a79586d0e9de`.
`benchmarks/vllm_moe_native.py` compiles that checkout's original CUDA
alignment, activation and reduction files with a small standalone registration
unit. The upstream files retain their Apache-2.0 notices and remain external
dependencies; they and the compiled binaries are not redistributed here.
The upstream license is [available in the pinned checkout](https://github.com/vllm-project/vllm/blob/4e304b4d1ed0037514073305d839a79586d0e9de/LICENSE).

These helpers isolate serving imports for a single-GPU, unquantized BF16
functional expert FFN measurement. They do not implement a vLLM engine or
replace GrainGEMM's production kernels. Source, binding, configuration and
native-library hashes are retained with the benchmark results.
