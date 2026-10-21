**Thor MoE FFN: fused INT8 gate/up, SwiGLU and quantization — 2026-10-21**

The experimental fused producer preserves the current INT8 results, but does not improve end-to-end latency: it is 1.4%–18.8% slower over the four tested batch sizes. The existing production dispatch remains the default. The experiment is SM110-only; SM120/GB10 code is unchanged.

Thor GPU frequency was **reduced to nominal 1.2 GHz (actual 1197 MHz)** and **EMC was locked at 2750 MHz**. All minimum, maximum and current frequency readings matched these settings. Configuration: 32 experts, hidden width H=1536, intermediate width I=1024, Top-4 routing, no shared experts. All methods use identical original BF16 inputs and weights, expert IDs, and FP32 routing scores rounded through BF16.

Uninstrumented full eager calls use CUDA-event spans, no CUDA Graph. Each method has two warm calls, followed by three rounds of three samples with balanced execution order and a common preallocated 256 MB L2 flush outside each timer. Reported latency is the median of the three round medians. Routing logits/top-k selection, one-time weight quantization/transformation, compilation, and reusable Mega workspace setup are outside timing; runtime allocations remain inside. Current INT8 includes routing validation and its CPU capacity read, input quantization/packing, both GEMMs, activation quantization and combine. Fused INT8 replaces only G1 plus activation quantization. FP8 includes direct BF16-to-FP8 input quantization, ID/score copies, output allocation, and Mega dispatch/G1/activation/quantization/G2/combine. No inter-GPU communication is used.

Throughput counts useful-token GEMM operations only: G1=4×T×R×H×I, G2=2×T×R×H×I, full FFN=6×T×R×H×I; padding and nonlinear/quantization arithmetic are excluded. Integer operations are reported as TOPS, FP8 operations as TFLOPS.

| Tokens | Current INT8 ms (TOPS) | Fused INT8 ms (TOPS) | Fused latency change | DeepInfer FP8 ms (TFLOPS) | FP8 speedup over current |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.546 (43.6) | 3.594 (43.0) | +1.4% | 2.393 (64.6) | 1.48× |
| 8192 | 5.838 (53.0) | 6.259 (49.4) | +7.2% | 4.139 (74.7) | 1.41× |
| 16384 | 10.070 (61.4) | 11.579 (53.4) | +15.0% | 7.422 (83.3) | 1.36× |
| 32768 | 19.297 (64.1) | 22.916 (54.0) | +18.8% | 14.041 (88.1) | 1.37× |

Source: [full benchmark JSON](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110.json) and [CSV](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110.csv). Maximum sampled temperature was 62.281°C; cooling begins at 62°C and the run aborts at 70°C.

The following stage tables come from a separate Nsight Systems profile: three complete calls per method and token count. Cells show **mean exclusive GPU activity milliseconds / share of that method’s mean outer CUDA-event span**, with useful throughput in parentheses for projection, fused projection, Mega and full FFN rows. These instrumented means differ from the uninstrumented medians above. No cross-stage GPU overlap was observed; exclusive activities plus outside-GPU residual add to each mean outer span, before rounding.

“Outside GPU activities” is the event span minus the union of recorded kernel/memcpy/memset intervals. It includes submission/synchronization gaps and event edges; it is not a direct measurement of CPU execution time. “Routing statistics transfer” is the observed GPU memcpy activity outside an explicit child stage, not the full CPU synchronization cost. Mega is one fused kernel; its gate/up, activation, down and combine costs cannot be separated from this trace.

**4096 tokens — profile means**

| Stage | Current INT8: ms / share (TOPS) | Fused INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0224 / 0.6% | 0.0214 / 0.6% | — |
| INT8 input quantization + packing | 0.2995 / 7.8% | 0.2993 / 7.7% | — |
| FP8 input quantization | — | — | 0.1712 / 6.9% |
| FP8 route ID/score copies | — | — | 0.0055 / 0.2% |
| Gate/up GEMM | 1.2935 / 33.5% (79.7) | — | — |
| SwiGLU + INT8 quantization | 0.5375 / 13.9% | — | — |
| Fused gate/up + SwiGLU + INT8 quantization | — | 1.9165 / 49.6% (53.8) | — |
| Down GEMM | 0.7769 / 20.1% (66.3) | 0.7730 / 20.0% (66.7) | — |
| Output combine | 0.3571 / 9.2% | 0.3553 / 9.2% | — |
| Mega MoE kernel | — | — | 2.1650 / 86.9% (71.4) |
| Routing statistics transfer | 0.0024 / 0.1% | 0.0021 / 0.1% | — |
| Outside GPU activities | 0.5726 / 14.8% | 0.4988 / 12.9% | 0.1493 / 6.0% |
| Full FFN, profiled | **3.8619 / 100% (40.0)** | **3.8662 / 100% (40.0)** | **2.4910 / 100% (62.1)** |

**8192 tokens — profile means**

| Stage | Current INT8: ms / share (TOPS) | Fused INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0381 / 0.6% | 0.0373 / 0.6% | — |
| INT8 input quantization + packing | 0.5497 / 9.2% | 0.5479 / 8.5% | — |
| FP8 input quantization | — | — | 0.2748 / 6.6% |
| FP8 route ID/score copies | — | — | 0.0078 / 0.2% |
| Gate/up GEMM | 2.0135 / 33.6% (102.4) | — | — |
| SwiGLU + INT8 quantization | 1.0563 / 17.6% | — | — |
| Fused gate/up + SwiGLU + INT8 quantization | — | 3.5215 / 54.7% (58.5) | — |
| Down GEMM | 1.1938 / 19.9% (86.3) | 1.1882 / 18.5% (86.8) | — |
| Output combine | 0.7437 / 12.4% | 0.7442 / 11.6% | — |
| Mega MoE kernel | — | — | 3.7844 / 91.0% (81.7) |
| Routing statistics transfer | 0.0015 / 0.0% | 0.0019 / 0.0% | — |
| Outside GPU activities | 0.4018 / 6.7% | 0.3984 / 6.2% | 0.0903 / 2.2% |
| Full FFN, profiled | **5.9984 / 100% (51.6)** | **6.4394 / 100% (48.0)** | **4.1573 / 100% (74.4)** |

**16384 tokens — profile means**

| Stage | Current INT8: ms / share (TOPS) | Fused INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0716 / 0.7% | 0.0712 / 0.6% | — |
| INT8 input quantization + packing | 1.0798 / 10.5% | 1.0820 / 9.2% | — |
| FP8 input quantization | — | — | 0.5041 / 6.8% |
| FP8 route ID/score copies | — | — | 0.0127 / 0.2% |
| Gate/up GEMM | 3.1420 / 30.7% (131.2) | — | — |
| SwiGLU + INT8 quantization | 2.0586 / 20.1% | — | — |
| Fused gate/up + SwiGLU + INT8 quantization | — | 6.7746 / 57.5% (60.9) | — |
| Down GEMM | 1.9510 / 19.0% (105.7) | 1.9521 / 16.6% (105.6) | — |
| Output combine | 1.4933 / 14.6% | 1.4962 / 12.7% | — |
| Mega MoE kernel | — | — | 6.8102 / 91.7% (90.8) |
| Routing statistics transfer | 0.0015 / 0.0% | 0.0015 / 0.0% | — |
| Outside GPU activities | 0.4478 / 4.4% | 0.4142 / 3.5% | 0.1011 / 1.4% |
| Full FFN, profiled | **10.2456 / 100% (60.4)** | **11.7918 / 100% (52.4)** | **7.4281 / 100% (83.3)** |

**32768 tokens — profile means**

| Stage | Current INT8: ms / share (TOPS) | Fused INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.2887 / 1.5% | 0.2846 / 1.2% | — |
| INT8 input quantization + packing | 2.1885 / 11.0% | 2.1705 / 9.4% | — |
| FP8 input quantization | — | — | 0.9679 / 6.9% |
| FP8 route ID/score copies | — | — | 0.0197 / 0.1% |
| Gate/up GEMM | 5.8282 / 29.3% (141.5) | — | — |
| SwiGLU + INT8 quantization | 4.1256 / 20.8% | — | — |
| Fused gate/up + SwiGLU + INT8 quantization | — | 13.4380 / 58.0% (61.4) | — |
| Down GEMM | 3.8703 / 19.5% (106.5) | 3.7732 / 16.3% (109.3) | — |
| Output combine | 3.0548 / 15.4% | 3.0498 / 13.2% | — |
| Mega MoE kernel | — | — | 12.9299 / 92.1% (95.7) |
| Routing statistics transfer | 0.0017 / 0.0% | 0.0018 / 0.0% | — |
| Outside GPU activities | 0.5239 / 2.6% | 0.4591 / 2.0% | 0.1145 / 0.8% |
| Full FFN, profiled | **19.8816 / 100% (62.2)** | **23.1771 / 100% (53.4)** | **14.0319 / 100% (88.2)** |

Profile source: [analysis JSON](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110_profile_analysis.json), [stage CSV](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110_profile_breakdown.csv), [call metadata](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110_profile_metadata.json), and [Nsight Systems trace](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110.nsys-rep). All 36 profiled calls have the expected stages, source hashes match, and the analysis reports no warnings or unclassified kernels.

The fused producer removes the BF16 gate/up intermediate write and read while retaining the BF16 rounding before FP32 SwiGLU. For useful rows, this removes 8×T×R×I bytes of logical tensor traffic:

| Tokens | Avoided logical BF16 write + read |
|---:|---:|
| 4096 | 128 MiB |
| 8192 | 256 MiB |
| 16384 | 512 MiB |
| 32768 | 1024 MiB |

These byte counts exclude padding and are not measured DRAM traffic or achieved bandwidth. The saved I/O does not compensate for the prototype’s slower fused computation: at 4096 tokens, separate G1 plus activation take 1.8310 ms of mean GPU activity versus 1.9165 ms fused; at 32768, they take 9.9538 ms versus 13.4380 ms. G2 is broadly unchanged. The candidate uses 199296 bytes of shared memory, 640 threads per CTA and 91 registers per thread, with zero compiler-reported register spills, and computes gate and up in serial MMA phases. Its resource footprint and scheduling are plausible optimization targets; this is an inference, not a measured attribution of the slowdown.

Correctness checks passed for 10 edge cases, each run twice: empty experts, one valid row, 63/64/65-row boundaries, partial tiles, long/short expert mixtures, zero input, zero weights, and zero gate channels. All four full-size cases also passed bitwise comparisons of complete INT8 intermediate values/scales including padding, complete BF16 FFN outputs, and repeated outputs before/after timing. Independent CPU quantized references checked two tokens per case. The FP8 input values/scales match the original input-cast helper bitwise, and expert counts match the supplied route IDs.

INT8 and FP8 are different numerical contracts: INT8 uses G256 with arbitrary FP32 scales and applies route scores after G2; FP8 uses E4M3 G32 with power-of-two UE8M0 scales, fast math, and scores before the intermediate quantization/G2. On the two-token common FP32-reference samples, relative L2 error is 2.26%–2.30% for INT8 and 6.28%–6.76% for FP8. This is a limited numerical check, not a model-accuracy result or an equal-accuracy comparison.

Memcheck completed the same 10 cases twice with full intermediate bit checks, exit code 0 and `ERROR SUMMARY: 0 errors`, using `--report-api-errors explicit` without disabling memory instrumentation. The first run with the default extended API reporting emitted one TileIR compiler-load API error at `cudaGetDeviceCount`; both original logs are retained. [Passing memcheck log](/workspace/artifacts/thor_moe_fusion_20261021/test_moe_fused_sm110_v2_memcheck_explicit.log); [initial log](/workspace/artifacts/thor_moe_fusion_20261021/test_moe_fused_sm110_v2_memcheck.log).

Provenance: grain-gemm HEAD `1651533ba5a498eca048594ae91f9dcb47425d3f` plus the recorded working-tree input-I/O optimization; DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`; vendored DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`. The DeepInfer/DeepGEMM FP8 reference uses three explicit host architecture-acceptance guard changes for real SM110, plus the separately verified direct input quantizer; it is an SM110 compatibility experiment, not an unmodified upstream Thor implementation. Device kernel arithmetic, layout and scheduling remain the original Mega implementation. The INT8 prototype preserves public weight/scale layouts without hidden weight prepacking.

Before/after source, package payload and native-library hashes are unchanged. Raw JSON/CSV, logs, temporary experiment sources, binaries and profile captures are stored outside Git at [thor_moe_fusion_20261021](/workspace/artifacts/thor_moe_fusion_20261021). The benchmark script is [bench_moe_fused_sm110.py](/workspace/artifacts/thor_moe_fusion_20261021/bench_moe_fused_sm110.py) (SHA256 `960bfbbd7a5e169ebd63e1ccb91fb1b73a7223f11190b1e49e484e21253caafa`); the candidate manifest is [moe_fused_sm110_v2_manifest.json](/workspace/artifacts/thor_moe_fusion_20261021/moe_fused_sm110_v2_manifest.json). Only this report belongs in the repository.
