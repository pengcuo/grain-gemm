# Static binary audit for the M256 investigation

CPU-only inspection of the existing native library; no CUDA module loading, GPU launch, ncu run, build, download, or grain-gemm repository edit was performed by this audit.

The compiled sm_121 kernels do not contain local-memory register spills or a floating-point tensor-multiply fallback. Every one of the 34 kernels (17 configurations, both grid traversals) has `STACK:0`, `LOCAL:0`, no `LDL`/`STL` SASS instruction, and only `IMMA.16832.S8.S8` as its MMA instruction. FP32 conversion, multiply, and FFMA instructions are present for the expected group-scale application; their presence is not a GEMM fallback. Static evidence rules out these compiled-code explanations, but does not measure stalls or establish the runtime bottleneck.

| Config | Traversal | Threads | Registers/thread | Nominal registers/CTA | Static shared B (cuobjdump) | Dynamic shared B (source launch) | Stack/local B | Static IMMA instructions |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 3 | 2D | 128 | 236 | 30208 | 1024 | 73728 | 0/0 | 64 |
| 3 | grouped | 128 | 246 | 31488 | 1024 | 73728 | 0/0 | 64 |
| 5 | 2D | 128 | 250 | 32000 | 1024 | 73728 | 0/0 | 64 |
| 5 | grouped | 128 | 248 | 31744 | 1024 | 73728 | 0/0 | 64 |
| 6 | 2D | 256 | 248 | 63488 | 1024 | 65536 | 0/0 | 64 |
| 6 | grouped | 256 | 248 | 63488 | 1024 | 65536 | 0/0 | 64 |
| 7 | 2D | 256 | 248 | 63488 | 1024 | 98304 | 0/0 | 64 |
| 7 | grouped | 256 | 248 | 63488 | 1024 | 98304 | 0/0 | 64 |
| 15 | 2D | 256 | 234 | 59904 | 1024 | 101376 | 0/0 | 64 |
| 15 | grouped | 256 | 242 | 61952 | 1024 | 101376 | 0/0 | 64 |

The register product is nominal, before any hardware allocation rounding. cfg7 and cfg6 both use 248 registers/thread and 256 threads; reducing cfg7 from three stages to two stages reduces source-requested dynamic shared memory by 32,768 B without reducing its register allocation. cfg3/cfg5 use roughly half the registers per CTA because they use 128 threads. High allocation is distinct from spilling; resident blocks and achieved occupancy require the CUDA occupancy query/runtime evidence.

The static `SHARED:1024` field must not be mistaken for total launch shared memory. Source-defined operand rings use `(BM+BN)*128*stages` bytes, and async-scale variants add `4*(BM+BN)*stages`. The launch passes `sizeof(SharedStorage<...>)` as dynamic shared memory. These amounts are source-derived, not read back from a CUDA function in this audit. The source checksum matches build.json. The CUDA ELF also contains a CUDA_RESERVED_SHARED section, but this audit does not assign a purpose to each byte of the 1,024 B resource field.

The mapping is decoded from demangled kernel template arguments, not guessed from order: Grouped, AsyncScale, StoreBits, CtaTiler BM/BN/BK, and the pipeline extent in ASmemLayout. It is matched to the native dispatch cases and cuda.CONFIGS. The first template Boolean selects 2D for K <= 4096 and grouped for K > 4096. Both versions were checked.

Relevant source locations:

- [src/grain_gemm/kernels/cuda.py](../../../src/grain_gemm/kernels/cuda.py#L12): tile/config registry.
- [src/grain_gemm/kernels/csrc/grain_cute.cu](../../../src/grain_gemm/kernels/csrc/grain_cute.cu#L55): shared storage and async scale-ring specialization.
- [src/grain_gemm/kernels/csrc/grain_cute.cu](../../../src/grain_gemm/kernels/csrc/grain_cute.cu#L337): INT32-to-FP32 group accumulation/scaling.
- [src/grain_gemm/kernels/csrc/grain_cute.cu](../../../src/grain_gemm/kernels/csrc/grain_cute.cu#L402): signed INT8 tensor MMA atom.
- [src/grain_gemm/kernels/csrc/grain_cute.cu](../../../src/grain_gemm/kernels/csrc/grain_cute.cu#L410): dynamic shared-memory size and launch.
- [src/grain_gemm/kernels/csrc/grain_cute.cu](../../../src/grain_gemm/kernels/csrc/grain_cute.cu#L437): config dispatch mapping.

Archived artifacts:

- [static-audit.json](static-audit.json): exact symbols, source/binary hashes, resource values and SASS opcode counts for all 34 variants.
- [static-audit.csv](static-audit.csv): compact resources for all variants.
- [static-audit-summary.txt](static-audit-summary.txt): summary of register, local-memory and opcode checks.
- [occupancy.json](occupancy.json): subsequent CUDA occupancy-query results for the measured binary.

Raw SASS dumps, extracted cubins and machine-local parsing scripts are not distributed in this archive. Line numbers in the original JSON refer to those historical dumps; the JSON is byte-preserved. Static instruction counts are not hardware utilization measurements.

Native library SHA256: `4c1940d5105732617cebb3b799a753a66565061d5e3440b2af8bd133e573c0ce`.

Source SHA256: `ff3707f9ae33075add88148fa680b768cccd13c748bf8defa1136ad48e448148` (matches build.json).
