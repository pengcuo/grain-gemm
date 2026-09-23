# Kernel design and native build

The public `grain_gemm.int8_gemm` operation keeps a separate INT32 partial sum for
each K-group and an FP32 sum across groups. Output defaults to BF16, with
conversion after FP32 accumulation. Set `output_dtype=torch.float32` or
`output_dtype=torch.float16` explicitly for FP32 or FP16 output; `backend="auto"`
uses Triton for these output types. Inputs are already quantized; there is no
hidden quantizer or weight packing. All timings and validation results are in
[Tests and benchmarks](../benchmarks/README.md).

## Implementations

The custom [Triton kernel](../src/grain_gemm/kernels/triton.py) makes group size,
strides and bounds compile-time parameters. It pipelines upcoming group loads,
uses Tensor Core INT8 dot products, and applies the two scales once per complete
or masked K-group. A grouped tile traversal improves L2 reuse. It accepts
positive-stride views and partial M/N/K tiles without copying the inputs.

The optional [CUDA/CuTe kernel](../src/grain_gemm/kernels/csrc/grain_cute.cu) adapts
the CUTLASS SM80 CuTe tutorial and specializes its pipeline for GB10. Each G256
consists of two K128 tiles accumulated into the **same INT32 fragment** using
`mma.sync.aligned.m16n8k32`. The complete group sum is converted to FP32 and
combined with the running FP32 output using `fmaf(partial, scale_a * scale_b,
output)`. Conversion to BF16 happens only in the epilogue. This preserves the
grouped operation without introducing an input quantizer or weight packing.

Operand loads use 16-byte `cp.async.cg` copies, which bypass L1, into swizzled
shared memory. `ldmatrix` loads feed the register pipeline and INT8 MMA. Copy
completion waits and CTA barriers protect each shared stage before reading or
reusing it. These are software-scaled INT8 kernels: the exposed block-scale MMA
formats do not provide this INT8 operation with arbitrary FP32 scales.
See NVIDIA's [MMA instruction reference](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
and [asynchronous copy reference](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

Seventeen configurations combine the following scale and output paths:

| Config IDs | Scale loads | BF16 output stores |
| --- | --- | --- |
| 0–7 | Prefetch row/column scales into registers before the group's first K128 tile | Redistribute fragments through shared memory, then issue coalesced 128-bit stores |
| 8–15 | Copy scales asynchronously into a shared ring alongside operand stages | Convert and store adjacent BF16 pairs with 32-bit stores |
| 16 | Same asynchronous scale ring as config 15 | Retain scalar 16-bit stores as a separate tuning candidate |

For the asynchronous path, the second K128 half reads its scales after the
previous iteration's wait and barrier, before that stage can be reused. The
128-bit epilogue drains all pending copies before reusing operand shared memory.
The [configuration list](../src/grain_gemm/kernels/cuda.py) records the exact
tile, warp count, stage count, scale path and store width for every ID.

The base tiles are 64×64, 128×64, 64×128 and 128×128, each with two or three
shared stages. The first three use four warps; 128×128 uses eight. On the tested
GB10, the driver reports 100 KiB shared memory per SM and a 99 KiB per-CTA opt-in
limit. A 128×128 K128 tile with three stages uses 96 KiB for operands; its scale
ring uses the remaining 3 KiB. Larger tiles or more stages must also fit the
register budget, so increasing either is not automatically beneficial.

K ≤ 4096 uses a two-dimensional grid; larger K uses groups of eight M tiles for
L2 locality, including partial final groups. TMA operand loading and unrolling
both K128 halves were also measured during tuning, but were not selected for
the final implementation. Results and reproduction commands remain in
[Tests and benchmarks](../benchmarks/README.md).

Implementations and toolchains can differ slightly in floating-point rounding.
The complete INT32 group reduction and FP32 cross-group accumulation remain
the common numerical contract.

## Build the optional GB10 CUDA backend

The alternative `backend="cutlass"` uses CUTLASS 3.x composition:
`GemmUniversalAdapter` launches a `GemmUniversal` specialization with a custom
G256 collective and a selectable epilogue. The collective adapts
the official SM80 multistage mainloop; a small device composition bridge adds
scale parameters and FP32 output fragments while retaining the standard
argument lowering and launch machinery. It is compiled for SM121 and uses
the same INT8 `mma.sync` and `cp.async` instruction families as the direct CuTe
backend. It does not invoke `grain_cute.cu`.

Its 32 configurations repeat eight tile/warp/stage choices in four families.
IDs 0–7 retain the standard FP32 shared exchange before 128-bit BF16 stores;
IDs 8–15 convert to BF16 before the shared exchange; IDs 16–23 combine
asynchronous scale staging with direct BF16 pair stores; IDs 24–31 retain
register scale prefetch with the same direct stores. The last family permits
the scale-loading change to be measured separately. All families retain
`GemmUniversalAdapter` and the G256/BF16/native-layout constraints below.
An independent `sm121_g256_cutlass.json` table selects configurations for
explicit CUTLASS calls on the measured M1024/M2048 model projection shapes;
it does not change the automatic backend policy.
This is a custom CUTLASS implementation, not an unchanged stock INT8 GEMM.
See [CUTLASS build and comparison](../benchmarks/cutlass.md).

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
BF16-output square shapes and exact rectangular `(M,N,K)` shapes. Each entry
stores the selected backend, the best measured CUDA configuration, and the best
measured Triton configuration.
`backend="cuda"` selects the stored CUDA candidate even when Triton wins that
shape; `backend="triton"` selects the stored Triton candidate. Auto dispatch uses
the overall winner. If native code is unavailable or its layout constraints are
not met, auto dispatch uses Triton. No search occurs during inference.
The table chooses the fastest candidate in the documented search, not a proven
global optimum; different toolchains or operating conditions can change rankings.

The measured rectangular entries currently select these CuTe configurations:

| M | N | K | CuTe config | Output tile |
|---:|---:|---:|---:|---|
| 256 | 1024 | 1536 | 3 | 128 × 64 |
| 256 | 1536 | 1024 | 5 | 64 × 128 |
| 256 | 2048 | 1536 | 7 | 128 × 128 |

The first two shapes benefit from smaller tiles that distribute work across
more SMs. The third retains the larger tile because the smaller tiles exceed
one batch of resident CTAs on GB10. These are exact shape matches, not a rule
for every M=256 shape. Each entry also retains its measured Triton fallback.
The top-level `tuning_report_sha256` identifies the square search; each
rectangular entry has a separate `measurement_sha256` for its timing record.
The optional `cutlass` field stores the independent collective backend's
measured configuration, identified by `cutlass_measurement_sha256`; it is used
by explicit `backend="cutlass"` selection. Auto continues to use `selected`.

Other SM121 shapes use a fixed Triton heuristic. Other SM80+ architectures,
group sizes and layouts use a conservative Triton configuration. Those paths
must be measured on their actual target devices before making speed claims;
H100 and Thor do not inherit the GB10 tuning table. Future specialized kernels
can be added to the same dispatch layer.

The native path requires SM121, G256, BF16 output, positive M/N multiples of 64,
and K a positive multiple of 256. A must be contiguous [M,K], and logical B[K,N]
must have contiguous [N,K] backing storage; both pointers must be 16-byte aligned.
FP32 scales must be contiguous with shapes [M,K/256] and [K/256,N]. Individual
configurations can require larger M/N multiples to match their tiles. Explicit
CUDA selection reports an error when these constraints or the native build are
unavailable; unmeasured compatible shapes use a fixed CUDA configuration.

Use `get_kernel_config(a, b, group_size=256, output_dtype=torch.bfloat16,
scale_a=scale_a, scale_b=scale_b, backend="auto")` to inspect the exact selection.
The implementation uses 32-bit relative element offsets and rejects tensors or
output shapes that exceed that range.
