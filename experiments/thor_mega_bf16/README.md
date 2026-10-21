# Thor Mega BF16 comparison

This optional four-way comparison reproduces the interface and numerical scopes
of the [measured Thor report](../../benchmarks/results/thor/moe_ffn_mega_bf16_emc2750.md):
selected Mega INT8, DeepInfer Mega FP8, DeepInfer Mega BF16 and compiled Cosmos
BF16. It does not modify production dispatch or GB10. This portable harness is
separate from the archived timing executable; the report identifies the exact
archived executable, measured sources, libraries and raw samples.

Use one Thor SM110 GPU with compatible PyTorch, Triton and CUDA13.2. The Cosmos
baseline requires `torch._grouped_mm`. Building DeepGEMM's host extension also
requires its normal setuptools/packaging dependencies and the `tree_sitter` and
`tree_sitter_cpp` modules used by its stub generator. The measured build used
tree-sitter0.26.0 and tree-sitter-cpp0.23.4. No model or distributed process group
is needed for the one-rank buffer path.

## Prepare the isolated DeepGEMM source/package

Use DeepInfer commit `8b16149741a4935e1dec8cea27853f273b374c5b` with its initialized
submodules. Keep all generated directories outside the GrainGEMM checkout:

```bash
thor_work_dir="$PWD/../thor-mega-bf16-work"
mkdir -p "$thor_work_dir"
git clone https://github.com/deepseek-ai/DeepInfer.git "$thor_work_dir/DeepInfer"
git -C "$thor_work_dir/DeepInfer" checkout 8b16149741a4935e1dec8cea27853f273b374c5b
git -C "$thor_work_dir/DeepInfer" submodule update --init --recursive
python experiments/thor_mega_bf16/build.py \
  --deepinfer-dir "$thor_work_dir/DeepInfer" \
  --source-dir "$thor_work_dir/source" --build-dir "$thor_work_dir/package" --build
```

The builder enforces the DeepGEMM/CUTLASS/deep_jit pins in
[THIRD_PARTY.md](THIRD_PARTY.md), requires clean tracked sources, and applies
exactly the four changes in [compatibility.patch](compatibility.patch).
It materializes external headers, copies the required deep_jit header payload
omitted by upstream setup, and verifies the resulting source/package hashes.
Only the host extension/package is built; this command executes no GPU kernel.
DeepGEMM compiles device kernels lazily during benchmark warmup on the real GPU.

Omit `--build` to prepare source only, or use `--dry-run` to inspect the plan
without copying/building. `--build-deps-dir` can supply an existing Python build
dependency directory; `--extra-include-dir` can supply supplemental host headers.
Source/build targets must be new, separate directories. Build logs and manifests
are written to the selected build directory. Do not rebuild a package already
loaded in a Python process; start a new process afterward.

## Run the comparison

First build the selected [INT8 experiment](../thor_interleaved_mega/README.md),
including its exact SiLU table variant. Provide that explicit library and a
pinned Cosmos checkout to the portable benchmark:

```bash
PYTHONPATH=src python experiments/thor_mega_bf16/benchmark.py \
  --deepinfer-dir "$thor_work_dir/DeepInfer" \
  --deepgemm-source-dir "$thor_work_dir/source" \
  --deepgemm-package-dir "$thor_work_dir/package/deep_gemm" \
  --cosmos-dir /path/to/cosmos-framework \
  --int8-library /path/to/int8-lut/libgrain_moe_mega.so \
  --tokens 4096 8192 16384 32768 --rounds 4 --samples 3 --warmup 2 \
  --output "$thor_work_dir/comparison.json"
```

Use `--smoke-only` for untimed correctness checks. An optional
`--before-package-dir` selects the frozen INT8 reference package; otherwise the
current public GrainGEMM path supplies the untimed reference. The actual package
and hashes are recorded. The measured Cosmos revision is
`4ed6244506e00bb270ed17d4532897f2aaa7bc16`.

The benchmark reads all six GPU/EMC min/max/current frequencies and requires
GPU1197MHz (reduced nominal1.2GHz) and EMC2750MHz. It never sets clocks. It cools
at62C to54C and aborts at70C. Full calls use eager CUDA-event spans, rotating
method order and a256MB cache flush outside each sample. Run this as the only
GPU workload; archive copies and other substantial memory traffic should also
be outside timing on Thor's shared memory system.

## Numerical and timing scope

All methods share original BF16 inputs/weights and route scores rounded through
BF16. Entry INT8/FP8 quantization, weight preparation, compilation and reusable
Mega workspace setup occur outside timing. Mega BF16 times input/ID/score copies,
output allocation and the full persistent kernel. Mega FP8 starts with resident
quantized input and times ID/score copies, output allocation and the full kernel.
INT8 retains validation, routing, the host capacity read, fresh allocation and
its separate weighted combine; Cosmos retains its original compiled operations.

Mega BF16/FP8 use `fast_math=True`, `activation_clamp=None`, SwiGLU and no shared
experts. Their scores multiply FP32 activation before the hidden rounding and
Down; GrainGEMM scores multiply BF16 Down output afterward. Cross-format bit
identity is therefore not required. BF16 correctness uses an independent
per-expert reference with BF16 Gate/Up, FP32 SwiGLU/score multiplication, one
BF16 hidden rounding, BF16 Down and FP32 slot-order reduction. Fast activation
math and accumulation order can differ slightly. These are synthetic numerical
checks, not a model-quality evaluation.

Fusion still materializes DeepGEMM's global hidden BF16/FP8 ring and BF16 output
slots. Complete-FFN throughput uses useful `6*T*TopK*H*I` operations, excluding
padding and non-GEMM arithmetic; INT8 uses TOPS and floating-point methods use
TFLOPS. A fused implementation is not automatically faster than a separate one.
