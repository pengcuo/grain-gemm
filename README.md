# GrainGEMM

INT8 activation × INT8 weight (W8A8) GEMM with scales grouped along the reduction dimension, K.

Python package: `grain_gemm`.

The inference-only [`moe_ffn` API](docs/moe_ffn.md) accepts INT8 activations plus
FP32 `scale_x` and composes routing/packing, both
expert projections, SwiGLU/intermediate quantization and weighted combination
into one call. It uses the same logical weight views and scale layouts as GEMM.

For physically adjacent gate/up channels, pass
[`gate_up_layout="interleaved"`](docs/moe_interleaved.md) and supply both
weight and scale output columns as `[g0,u0,g1,u1,...]`. Thor uses the independent
`tools/build_moe_interleaved.py --cutlass-dir PATH` build for that layout.

On Thor SM110, the native Gate/Up kernel reads original `x_int8[T,H]` and
`scale_x[T,H/256]` through sorted route indices. It gathers directly into shared
memory and fuses Gate/Up, BF16 rounding, SwiGLU and G256 intermediate quantization.
It allocates neither an expert-expanded input/scale buffer nor a BF16 Gate/Up
workspace. Build the native down projection and original-token kernel:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /path/to/cutlass
python tools/build_moe_gather.py --cutlass-dir /path/to/cutlass
```

Auto selects native Thor execution for dense-enough BF16-output routing when the
builds are available and original x is 16-byte aligned. Other auto cases use
ragged Triton, which also reads original token rows directly. Captured explicit
Thor native calls support this gather path; captured auto calls retain Triton.
Down and output combination remain separate stages. Public weight/scale layouts,
single-GEMM implementations and SM120/GB10 native code are unchanged.

The [original-token benchmark](benchmarks/results/thor/moe_ffn_original_x_emc2750.md)
compares this path with the preceding INT8-input packed implementation and
DeepInfer FP8 at reduced GPU 1.2 GHz (actual 1197 MHz) and EMC 2750 MHz.
The [prequantized-input comparison](benchmarks/results/thor/moe_ffn_prequantized_emc2750.md)
and [cached two-SM benchmark](benchmarks/results/thor/moe_ffn_cached_2sm_emc2750.md)
retain their earlier implementations and documented input boundaries.

The explicitly selected [interleaved INT8 Mega experiment](experiments/thor_interleaved_mega/README.md)
keeps hidden quantization buffers on chip. The separate
[Mega BF16 comparison](experiments/thor_mega_bf16/README.md) provides a reproducible
four-way INT8/FP8/BF16/Cosmos benchmark and
[performance report](benchmarks/results/thor/moe_ffn_mega_bf16_emc2750.md).
These experiments do not replace production automatic dispatch.

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

GrainGEMM includes a custom Triton kernel and optional native CUDA/CuTe and CUTLASS kernels, with measured G256 dispatch tables for **GB10 (SM121)**. The public API selects a backend and launch configuration by architecture, shape, dtype, and layout. Thor (SM110) has a native `tcgen05` INT8/Tensor Memory implementation and a separate measured dispatch table; see [Thor validation and performance](benchmarks/results/thor/README.md). A100 and H100 use the portable Triton path pending validation and dedicated tuning on those GPUs. **SM120 native builds are experimental and require explicit backend selection**; RTX 50-series correctness and performance have not yet been validated, and `auto` continues to use Triton on SM120.

Native sources are organized by architecture under
[`kernels/csrc/sm12x`](src/grain_gemm/kernels/csrc/sm12x), while measured tuning
tables use GPU-specific names such as `gb10_sm121_g256.json`. Future H100 specializations will live in separate architecture directories. Thor sources
are under `kernels/csrc/sm110/`. See [source organization](docs/kernel_design.md#source-organization).

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

`backend="auto"` uses the measured GB10 or Thor configuration where applicable. The
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

### Thor SM110 native build

Use the same `int8_gemm` and `get_kernel_config` API as GB10. Scale layouts are
unchanged: contiguous FP32 `scale_a[M,K/256]` and `scale_b[K/256,N]`; A is
row-major `[M,K]` and logical B `[K,N]` is a view of row-major weights `[N,K]`.
There is no additional scale packing, input quantization or hidden copy.

For `Q = K/256`, `scale_a` has stride `(Q, 1)`: its K-group dimension is
contiguous. `scale_b` has stride `(N, 1)`: its N dimension is contiguous.
The Thor CuTe and CUTLASS paths read these same layouts directly.

```bash
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_110
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_110
```

The CUDA build uses `sm_110a` instructions and `-DNDEBUG` while runtime metadata
identifies SM110. Explicit `backend="cuda"` selects the independent Thor CuTe
implementation. Its new warp-specialized candidates are:

| CuTe config | Output tile | Operand stages | SMs per tile |
| --- | --- | --- | --- |
| 26 | 256 × 256 | 3 | 2 |
| 27 | 256 × 256 | 2 | 2 |
| 28 | 128 × 128 | 3 | 1 |
| 29 | 128 × 256 | 2 | 1 |

Two-SM configs 26/27 accept M multiples of 256 and N multiples of 128,
including a final 128-column half-tile. One-SM configs 28/29 require complete
tiles of their listed sizes.

Each new candidate uses 384 threads per CTA: four producer warps feed TMA/UMMA
and scale loads, while eight consumer warps promote INT32 group results into
FP32. Two TMEM buffers and three prefetched register fragments overlap
operand transfers, MMA and scaling. The FP32 scale product feeds
`fmaf(float(partial), scale_a * scale_b, total)`; BF16 conversion occurs once
after the final group. Other incompatible tile or TMA alignment cases use the
existing warp-MMA config 0 fallback. Calls use the current PyTorch CUDA stream, rebuild
pointer-dependent descriptors per launch, and support CUDA Graph capture
without caching tensor pointers.

The CuTe candidates apply the pipeline ideas from the Thor CUTLASS path through
CuTe primitives; they execute their own kernels, independently of CUTLASS
configs 32/33. `backend="cutlass"` selects the separately built collective,
including a two-SM
256×256×256 pipeline with eight warps applying the FP32 group scales. It reads
the public scale layouts directly. The launcher rebuilds parameters on each
invocation and supports CUDA Graph capture and concurrent streams. Its CLC
scheduler requires no scratch workspace; any requested scratch is managed by PyTorch.
`backend="auto"` uses the Thor table for measured shapes, including CUTLASS
when its measured advantage justifies selection. It retains measured CUDA or
Triton fallbacks when CUTLASS is unavailable. GB10 tuning entries are never used on Thor.
For unmeasured shapes, explicit `backend="cuda"` starts with config 27 when
M/N are both multiples of 256, config 28 when both are multiples of 128,
and config 17 otherwise. Unmeasured `backend="auto"` shapes retain Triton.
Group sizes 32/64/128, FP16/FP32 outputs, K tails and other positive-stride layouts
retain the public Triton path. Native calls have the same G256/BF16/aligned-layout
constraints as GB10. See [Thor instructions and measured scope](benchmarks/results/thor/README.md).
SM120/GB10 sources, configuration IDs and build flags remain unchanged.

### Experimental SM120 native build

To try the existing native kernels on an SM120 device, build the desired backend
with the explicit target:

```bash
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_120
python tools/build_cutlass.py --cutlass-dir /path/to/cutlass --arch sm_120
```

Then pass `backend="cuda"` for CuTe or `backend="cutlass"` to `int8_gemm`.
The native G256, BF16 and layout constraints above still apply. These builds
use fixed compatible defaults instead of the GB10 tuning tables;
`backend="auto"` stays on Triton for SM120 until measurements justify native
dispatch. The default build target remains `sm_121`.

Each backend installs one target at a time; rebuilding it replaces its previous
library. Runtime requires the build's GPU target to match the device. Builds
can run without a GPU, but the shared library must also match the execution
machine's CPU architecture and host ABI; a GB10 ARM64 host build is not an
x86-64 desktop library. See [native build requirements](docs/kernel_design.md)
and [CUTLASS build details](benchmarks/cutlass.md#build-and-use).

RTX 50-series runtime correctness and performance remain unvalidated. Published
GB10 benchmark results and experimental measurement scripts retain their
GB10-only scope.

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
