# CUTLASS experiments

**GB10-only: SM121 / compute capability 12.1.** All performance results,
configuration choices and CuTe/CUTLASS rankings from these experiments apply
only to NVIDIA GB10. Thor / Jetson T5000 uses SM110 / compute capability 11.0
and requires independent implementation, tuning and validation. This boundary
does not restrict GrainGEMM's future multi-architecture support.

This directory preserves the independent CUTLASS G256 experiments used to investigate the performance difference from direct CuTe C++. The experiment libraries are loaded explicitly through `ctypes`; they do not participate in `backend="auto"` or replace the production libraries. The existing explicit production CUTLASS backend remains separate.

The [measured results and analysis](../../benchmarks/results/cutlass/residual-gap/) include the three original paired runs and the M=4096 extension, with raw rounds, TOPS tables, static instruction analysis, and validation records. These are fixed-configuration comparisons on NVIDIA GB10, not autotuning results for every shape or claims about other GPUs.

The earlier [32-configuration CUTLASS optimization experiment](optimization/README.md) preserves its own sources and reproduction entry points for the 18 M=1024/2048 model projection shapes.

## Preserved sources

| Directory | Experiment | Difference from the matched baseline |
|---|---|---|
| `variants/baseline/` | Matched control | Original CUTLASS computation, reduced translation unit containing configurations 11, 15, and 23; generated instructions matched the production library on the measured compiler. |
| `variants/no_direct_tail_barrier/` | A | Skip the final async wait and block barrier only for direct BF16 stores. Keep synchronization when shared memory is reused. |
| `variants/bf16_uint4_i32_address/` | B | Use 32-bit linear address calculation for shared-memory BF16 writeback. |
| `variants/bf16_fixed_store_loop/` | C | Give the BF16 shared-memory writeback loop a compile-time trip count. |
| `variants/late_fp32_accumulator_clear/` | D | Move FP32 accumulator initialization after operand prefetch. Generated identical instructions to the baseline in the measured build; no separate paired timing was necessary. |
| `variants/minimum_k_guard/` | E | Add device-side `if (K < 256) return;`. Valid inputs already satisfy this condition through host validation. |
| `variants/ungrouped_specialization/` | F | Remove the runtime grouped-scheduling branch, and reject K > 4096 at the launch boundary. |

The variants are independent changes, not successive optimizations. In particular, E and A are not combined. All `.cu` and `.hpp` files under `variants/` are byte-for-byte copies of the measured sources. Their original manifests retain source/library checksums, compiler version, and original build commands.

`measured_controls/cute/` and `measured_controls/cutlass/` freeze the original direct CuTe and 32-configuration CUTLASS control sources so future production changes do not silently change these comparisons. Their configuration numbers are separate: the matched pairs are CuTe 3 / CUTLASS 11 (128×64, 3 stages), CuTe 7 / CUTLASS 15 (128×128, 3 stages), and CuTe 15 / CUTLASS 23 (128×128, 3 stages, async scales and direct stores).

`archive/` retains the exact original measurement, reference, correctness, and M=4096 driver scripts. Absolute paths inside those archived scripts and the historical manifests are provenance from the original machine; they are not required by the portable entry points in this directory. `source_manifest.json` records all frozen source hashes.

The portable `paired_benchmark.py` changes path/CLI plumbing, adds input and device checks and refuses to overwrite case files. Its timing loop and statistical calculation retain the original method. `reference.py` and `correctness_fixture.py` contain the unchanged numerical helper functions from the archived scripts. New measurements record the new script hashes; historical result JSON remains unchanged and refers to the archived scripts.

GPU measurement and validation entry points require both a device name
containing `GB10` and compute capability `(12, 1)`, and reject all other GPUs.
CPU-only rendering, source/data audits and command previews do not require a
GPU. The build entry point only accepts `sm_121`; changing the target requires
a separate architecture implementation and validation effort.

## Build

Run commands from the repository root. The original measurements used CUDA 13.0.88, the NVIDIA GB10 (`sm_121`), and CUTLASS revision `098de2a652cf8f00fd70b2df54051c7eccbb855a`. Use a CUDA-enabled PyTorch installation. Other compiler or CUTLASS versions may produce different instructions and timings, and binary hashes can differ with build paths/options.

```bash
git clone https://github.com/NVIDIA/cutlass.git /path/to/cutlass
git -C /path/to/cutlass checkout 098de2a652cf8f00fd70b2df54051c7eccbb855a
python experiments/cutlass/build.py --cutlass-dir /path/to/cutlass
```

This builds both frozen controls and all seven variants under the ignored `experiments/cutlass/build/` directory, with source/compiler/library metadata. It uses the compiler options of `tools/build_cuda.py` and `tools/build_cutlass.py`; it does not write into `src/grain_gemm/kernels/_native/`. To compile only the controls and the two main ablations:

```bash
python experiments/cutlass/build.py --cutlass-dir /path/to/cutlass \
  --target cute_control cutlass_control baseline minimum_k_guard no_direct_tail_barrier
```

`--arch` accepts only `sm_121`; other targets are rejected during argument
parsing. `--nvcc` and `--build-dir` remain configurable. `--dry-run` prints the
build commands without compiling. Cross-compilation itself does not require a
GPU; measurement and validation still require GB10.

## Correctness and sanitizer

```bash
python experiments/cutlass/validate_variants.py
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python experiments/cutlass/validate_variants.py
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python experiments/cutlass/validate_variants.py
```

These check every output element for four small shapes, covering INT8 endpoints, row/column/group-varying positive, negative, and zero scales, pipeline startup/wrap, and grouped scheduling. Variant F must reject K=4352. Supply variant names after the script to validate a subset, for example `baseline minimum_k_guard no_direct_tail_barrier`. The historical records cover the tested variants; D was checked through identical generated instructions and did not receive an independent GPU correctness/timing run.

## Paired benchmark

```bash
python experiments/cutlass/run.py --suite m4096 --output experiments/cutlass/runs/m4096
python experiments/cutlass/summarize.py --input experiments/cutlass/runs/m4096 \
  --output experiments/cutlass/runs/m4096-summary
```

`--suite run1`, `run2`, or `run3` reproduces the original M=1024/2048 shapes and variant ordering. The default is 60 rounds. `--dry-run` prints commands without loading CUDA. Use a fresh output directory for each repetition; historical results are not overwritten.

For one explicit shape/configuration pair, or to compare a different built library:

```bash
python experiments/cutlass/paired_benchmark.py \
  --case 4096 1536 1024 7 15 --rounds 60 \
  --output experiments/cutlass/runs/custom \
  --variant cutlass_matched=experiments/cutlass/build/baseline/libgrain_cutlass.so \
  --variant cutlass_minimum_k=experiments/cutlass/build/minimum_k_guard/libgrain_cutlass.so
```

The defaults load the frozen controls from `build/`. Override them with `--cute-library` and `--cutlass-library`. Custom variants use the CUTLASS C ABI and must support the selected configuration.

Inputs are prequantized INT8 A/B with G=256, FP32 scales/accumulation, and BF16 output. A is row-major `[M,K]`; B is column-major `[K,N]`, constructed as a view of contiguous `[N,K]` storage. Timing includes group scaling and output conversion, excludes quantization/compilation/host dispatch, and uses preallocated output, repeated inputs, and warm caches. All variants use the same tensors and graph node count for each shape. Each round rotates and periodically reverses variant order, with three event-timed replays. TOPS counts `2*M*N*K` integer operations divided by elapsed time. The benchmark checks sampled 32×32 outputs plus full-output finiteness before timing and after graph replay; this sampled check does not replace full-output correctness tests.

The reports use the median of per-round medians. Paired ratios are computed within each round and can differ from the ratio of two overall medians. Bootstrap intervals describe the recorded run; adjacent rounds can be correlated, and they are not independent-machine confidence intervals. Runs 1 and 3 balance six paths exactly over 60 rounds; M=4096 balances five paths; run 2 has seven paths and therefore 8 or 9 appearances per execution position, also audited over its first 56 balanced rounds.

## CPU-only source and data checks

```bash
python experiments/cutlass/verify_sources.py \
  --results-dir benchmarks/results/cutlass/residual-gap
python experiments/cutlass/summarize.py \
  --input benchmarks/results/cutlass/residual-gap/paired_m4096 \
  --output experiments/cutlass/runs/published-m4096-summary
```

These commands require no GPU. Source checks validate the historical JSON against the untouched archived measurement/reference scripts. Rebuilt libraries are recorded by hash but need not be byte-identical to the original binaries, which are not committed.

The experiments support a code-generation explanation for the improvement from E on the tested 128×128 tile projections: both initialization and mainloop/control scheduling changed. They do not prove that all improvement comes from fewer clearing instructions or that CUTLASS/CuTe is universally faster. Nsight Compute returned `ERR_NVGPUCTRPERM` on the measurement machine; no hardware-counter-based stall explanation was established.

The preserved CUTLASS-derived sources retain their upstream license notices. See the repository [third-party notices](../../THIRD_PARTY_NOTICES.md).
