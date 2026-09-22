# Kernel design and native build

The public `grain_gemm.int8_gemm` operation keeps a separate INT32 partial sum for
each K-group and an FP32 sum across groups. Inputs are already quantized; there
is no hidden quantizer or weight packing. All timings and validation results are
in [Tests and benchmarks](../benchmarks/README.md).

## Implementations

The custom [Triton kernel](../src/grain_gemm/kernels/triton.py) makes group size,
strides and bounds compile-time parameters. It pipelines upcoming group loads,
uses Tensor Core INT8 dot products, and applies the two scales once per complete
or masked K-group. A grouped tile traversal improves L2 reuse. It accepts
positive-stride views and partial M/N/K tiles without copying the inputs.

The optional [CUDA/CuTe kernel](../src/grain_gemm/kernels/csrc/grain_cute.cu) adapts
the CUTLASS SM80 CuTe tutorial. It uses `mma.sync` INT8 instructions and explicit
`cp.async` shared-memory/register pipelines. Each G256 consists of two K128
tiles accumulated into the **same INT32 fragment** before scaling. An FP32
fragment holds the running output. Eight tile/stage configurations are compiled;
small K uses a two-dimensional grid, while K > 4096 uses groups of eight M tiles
for L2 locality. Output is BF16. This is software group scaling, not a native
MXINT8 instruction.

The two implementations may differ slightly in FP32 rounding because multiply
association and fused multiply-add differ. Both retain the complete INT32 group
reduction and FP32 cross-group accumulation.

## Build the optional GB10 CUDA backend

Use a source checkout (or unpacked source distribution), CUDA 13.0 or newer with
`sm_121` support, a compatible host C++ compiler, and the pinned CUTLASS headers:

```bash
python -m pip install -e ".[runtime]"
git clone https://github.com/NVIDIA/cutlass.git /path/to/cutlass
git -C /path/to/cutlass checkout 098de2a652cf8f00fd70b2df54051c7eccbb855a
python tools/build_cuda.py --cutlass-dir /path/to/cutlass --arch sm_121
```

Building is explicit: importing or calling GrainGEMM does not download headers
or invoke a compiler for the native backend. The build writes a local shared
library and metadata to `src/grain_gemm/kernels/_native/`; these are excluded from
Git and distributions. Rebuild after changing the CUDA source. The loader checks
the source hash and target architecture before using the library.

The launcher obtains the current PyTorch CUDA stream on every call, including
CUDA Graph capture. Warm up the desired shape before capture. No PyTorch C++ ABI
extension is required. See [third-party notices](../THIRD_PARTY_NOTICES.md) for
CUTLASS attribution and licensing.

## Dispatch and limits

`backend="auto"` consults the checked-in
[SM121 G256 table](../src/grain_gemm/kernels/configs/sm121_g256.json) for measured
BF16-output square shapes. The table stores both the selected backend and the
best measured Triton fallback. If native code is unavailable or the scale layout
is incompatible, the Triton fallback is used. No search occurs during inference.
The table chooses the fastest candidate in the documented search, not a proven
global optimum; different toolchains or operating conditions can change rankings.

Other SM121 shapes use a fixed Triton heuristic. Other SM80+ architectures,
group sizes and layouts use a conservative Triton configuration. Those paths
must be measured on their actual target devices before making speed claims;
H100 and Thor do not inherit the GB10 tuning table. Future specialized kernels
can be added to the same dispatch layer.

Use `get_kernel_config(a, b, group_size=256, output_dtype=torch.bfloat16,
scale_a=scale_a, scale_b=scale_b)` to inspect the exact selection. Force
`backend="triton"` for the portable implementation, or `backend="cuda"` to
require the native implementation and receive an error if its constraints or
build are unavailable. The implementation currently uses 32-bit relative element
offsets and rejects tensors/output shapes that exceed that range.
