**Thor INT8 MoE FFN: selective gate/up fusion — 2026-10-21**

The updated public `moe_ffn(..., backend="auto")` reduces full-FFN latency by **4.5% at 4096 tokens**, **6.0% at 8192**, and **3.9% at 32768**; 16384 is effectively unchanged. Gate/up, SwiGLU and activation quantization are fused at 4096/8192 only. Routing/input and projection tuning also contribute, so these are whole-change speedups, not isolated fusion speedups. DeepInfer FP8 remains **1.33–1.43× faster** in this test.

GPU frequency was **reduced to nominal 1.2 GHz, actual 1197 MHz**. **EMC was locked at 2750 MHz**. Minimum, maximum and current readings matched both locks throughout. Model: 32 experts, H=1536, I=1024, Top-4, no shared experts; identical original BF16 tokens/weights, expert IDs and FP32 routing scores rounded through BF16 for every method.

| Tokens | Frozen pre-change INT8: ms (TOPS) | Current INT8: ms (TOPS) | Latency reduction | DeepInfer FP8: ms (TFLOPS) |
|---:|---:|---:|---:|---:|
| 4096 | 3.567 (43.3) | **3.406 (45.4)** | **4.5%** | 2.388 (64.8) |
| 8192 | 5.840 (52.9) | **5.489 (56.3)** | **6.0%** | 4.133 (74.8) |
| 16384 | 10.064 (61.5) | 10.059 (61.5) | ≈0% | 7.400 (83.6) |
| 32768 | 19.303 (64.1) | **18.552 (66.7)** | **3.9%** | 13.962 (88.6) |

The baseline is a **frozen pre-change public package**, based on `1651533ba5a498eca048594ae91f9dcb47425d3f` plus its then-existing uncommitted input-I/O optimization. It is not a pristine checkout of that commit. Both INT8 columns call the actual public auto API. Raw results: [JSON](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final.json), [CSV](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final.csv), [log](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final.log).

Each method has two warm calls, then three rounds of three samples with balanced method order and a common preallocated 256 MB L2 flush outside each timer. Latency is the median of the three round medians. CUDA events span the complete eager call, including public validation, routing and CPU capacity synchronization, input quantization/packing, both projections, activation quantization, combination and runtime allocation. CUDA Graphs are not used. Router logits/top-k selection, one-time weight quantization/transformation, compilation and reusable Mega workspace setup are excluded. FP8 includes BF16-to-FP8 input quantization, route ID/score copies, output allocation and Mega execution. No inter-GPU communication is used. Maximum sampled temperature was 62.343°C; cooling starts at 62°C and execution aborts at 70°C.

Throughput uses useful-token projection operations: gate/up `4*T*R*H*I`, down `2*T*R*H*I`, full FFN `6*T*R*H*I`. Padding and nonlinear/quantization arithmetic are excluded. INT8 is reported in TOPS; FP8 in TFLOPS. These are complete-FFN rates, not bare GEMM rates.

The enabled SM110 kernel keeps BF16-rounded gate/up values in registers, computes FP32 SwiGLU and writes G256 INT8 activations/scales directly. The public API, transposed Linear weight views, contiguous scale layouts and rounding points remain unchanged. The full FFN still has separate routing, input packing, down GEMM and combine kernels. Other architectures, including SM120/GB10, retain their implementations.

| Tokens | Measured expert capacity C | Removed BF16 `[32,C,2048]` workspace | Logical GU write + valid activation read avoided |
|---:|---:|---:|---:|
| 4096 | 640 | 80 MiB | 144 MiB |
| 8192 | 1152 | 144 MiB | 272 MiB |

The logical I/O column counts the full padded GEMM output write plus the activation stage's useful-row read. These are tensor byte counts, **not measured DRAM traffic or bandwidth**. The implementation is a selective G1 fusion, not a complete single-kernel INT8 Mega MoE. The 16384/32768 cases retain separate native G1 and activation kernels; the 32768 improvement comes from routing/input and native tile choices. The optional library and fallback behavior are described in [the API documentation](../../../docs/moe_ffn.md).

All four sizes passed complete before/current bit comparisons of INT8 G2 inputs, FP32 intermediate scales, padding and final BF16 output, with repeated checks after timing. Two-token independent CPU quantized references matched the INT8 outputs exactly in these cases. Related regression tests passed **186 tests with 2 skips** across the suite and a targeted native-fallback test rerun, covering API shapes/strides, scales, masks, duplicate/invalid routes, empty experts, row boundaries, streams, replay and fallback behavior. Compute Sanitizer memcheck passed 10 edge cases twice with complete intermediate bit checks and **0 errors**: [log](/workspace/artifacts/thor_moe_mega_int8_20261021/verification/test_moe_fused_installed_memcheck.log).

DeepInfer uses its single-card FP8 Mega MoE from DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b` / vendored DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`, with three host architecture-acceptance guard changes for SM110 and a separately bit-verified direct input quantizer. This is an SM110 compatibility run, not an unchanged upstream Thor implementation; device kernel arithmetic/layout/scheduling remain unchanged. FP8 is E4M3 G32 with UE8M0 scales, `fast_math=True`, and route scores before intermediate quantization/G2. INT8 uses G256 FP32 scales and scores after BF16-rounded G2. Their numerical contracts differ; this is a performance reference, not an equal-accuracy or model-quality claim.

The following stage tables use **three separately profiled calls per method and token count**, not the nine-sample uninstrumented medians above. Each cell is **mean exclusive GPU activity ms / percentage of mean outer event span**. Projection parentheses are useful TOPS; fused G1's equivalent useful G1 TOPS includes SwiGLU/quantization time. FP8 Mega parentheses are TFLOPS for both projections together. No cross-stage overlap was observed; stage means plus the outside-GPU residual add to each mean total before rounding.

The routing-statistics D2H row contains the 24 small memcpy activities classified as `unclassified_sync`; it does not measure their full host synchronization cost. “Outside GPU activities” is event time minus the union of recorded kernels/memcpy/memset. It includes submission/synchronization gaps and event edges, and is not a direct measurement of CPU execution time. Mega remains one inseparable MoE kernel in this profile.

**4096 tokens — profile means**

| Stage | Frozen pre-change INT8: ms / share (TOPS) | Current INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0212 / 0.5% | 0.0206 / 0.6% | — |
| INT8 input quantization + packing | 0.2997 / 7.7% | 0.2978 / 8.4% | — |
| FP8 input quantization | — | — | 0.1694 / 7.0% |
| FP8 route ID/score copies | — | — | 0.0055 / 0.2% |
| Gate/up GEMM | 1.2969 / 33.5% (79.5) | — | — |
| SwiGLU + INT8 quantization | 0.5437 / 14.0% | — | — |
| Fused G1 + SwiGLU + INT8 quantization | — | 1.7642 / 49.6% (58.4) | — |
| Down GEMM | 0.7763 / 20.0% (66.4) | 0.7337 / 20.6% (70.2) | — |
| Output combine | 0.3563 / 9.2% | 0.3481 / 9.8% | — |
| Mega MoE kernel | — | — | 2.1511 / 88.5% (71.9) |
| Routing statistics D2H transfer | 0.0024 / 0.1% | 0.0020 / 0.1% | — |
| Outside GPU activities | 0.5777 / 14.9% | 0.3903 / 11.0% | 0.1047 / 4.3% |
| Full FFN, profiled | **3.8741 / 100% (39.9)** | **3.5567 / 100% (43.5)** | **2.4308 / 100% (63.6)** |

**8192 tokens — profile means**

| Stage | Frozen pre-change INT8: ms / share (TOPS) | Current INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0371 / 0.6% | 0.0344 / 0.6% | — |
| INT8 input quantization + packing | 0.5526 / 9.3% | 0.5493 / 9.7% | — |
| FP8 input quantization | — | — | 0.2765 / 6.7% |
| FP8 route ID/score copies | — | — | 0.0075 / 0.2% |
| Gate/up GEMM | 2.0076 / 33.7% (102.7) | — | — |
| SwiGLU + INT8 quantization | 1.0605 / 17.8% | — | — |
| Fused G1 + SwiGLU + INT8 quantization | — | 2.7045 / 47.9% (76.2) | — |
| Down GEMM | 1.2045 / 20.2% (85.6) | 1.2016 / 21.3% (85.8) | — |
| Output combine | 0.7494 / 12.6% | 0.7434 / 13.2% | — |
| Mega MoE kernel | — | — | 3.7901 / 91.4% (81.6) |
| Routing statistics D2H transfer | 0.0014 / 0.0% | 0.0021 / 0.0% | — |
| Outside GPU activities | 0.3521 / 5.9% | 0.4057 / 7.2% | 0.0705 / 1.7% |
| Full FFN, profiled | **5.9654 / 100% (51.8)** | **5.6410 / 100% (54.8)** | **4.1446 / 100% (74.6)** |

**16384 tokens — profile means**

| Stage | Frozen pre-change INT8: ms / share (TOPS) | Current INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.0726 / 0.7% | 0.0649 / 0.6% | — |
| INT8 input quantization + packing | 1.0774 / 10.5% | 1.0754 / 10.4% | — |
| FP8 input quantization | — | — | 0.5057 / 6.7% |
| FP8 route ID/score copies | — | — | 0.0126 / 0.2% |
| Gate/up GEMM | 3.1500 / 30.7% (130.9) | 3.1530 / 30.6% (130.8) | — |
| SwiGLU + INT8 quantization | 2.0559 / 20.1% | 2.0579 / 20.0% | — |
| Down GEMM | 1.9470 / 19.0% (105.9) | 1.9473 / 18.9% (105.9) | — |
| Output combine | 1.5039 / 14.7% | 1.5053 / 14.6% | — |
| Mega MoE kernel | — | — | 6.8614 / 91.5% (90.1) |
| Routing statistics D2H transfer | 0.0015 / 0.0% | 0.0015 / 0.0% | — |
| Outside GPU activities | 0.4392 / 4.3% | 0.4941 / 4.8% | 0.1208 / 1.6% |
| Full FFN, profiled | **10.2477 / 100% (60.4)** | **10.2994 / 100% (60.0)** | **7.5004 / 100% (82.5)** |

**32768 tokens — profile means**

| Stage | Frozen pre-change INT8: ms / share (TOPS) | Current INT8: ms / share (TOPS) | DeepInfer FP8: ms / share (TFLOPS) |
|---|---:|---:|---:|
| Routing/counting | 0.2886 / 1.5% | 0.1251 / 0.7% | — |
| INT8 input quantization + packing | 2.1621 / 10.9% | 2.0325 / 10.8% | — |
| FP8 input quantization | — | — | 0.9790 / 7.0% |
| FP8 route ID/score copies | — | — | 0.0191 / 0.1% |
| Gate/up GEMM | 5.7237 / 28.9% (144.1) | 5.4444 / 29.0% (151.5) | — |
| SwiGLU + INT8 quantization | 4.1145 / 20.8% | 4.1151 / 21.9% | — |
| Down GEMM | 3.9807 / 20.1% (103.6) | 3.5720 / 19.0% (115.4) | — |
| Output combine | 2.9992 / 15.2% | 3.0306 / 16.2% | — |
| Mega MoE kernel | — | — | 12.9176 / 92.0% (95.8) |
| Routing statistics D2H transfer | 0.0017 / 0.0% | 0.0016 / 0.0% | — |
| Outside GPU activities | 0.5057 / 2.6% | 0.4332 / 2.3% | 0.1222 / 0.9% |
| Full FFN, profiled | **19.7762 / 100% (62.5)** | **18.7544 / 100% (66.0)** | **14.0379 / 100% (88.1)** |

At 4096 tokens, separate G1 plus activation take 1.8406 ms of mean GPU activity versus 1.7642 ms fused; at 8192, 3.0681 ms versus 2.7045 ms. At 32768 the retained native G1/G2 improve from 5.7237/3.9807 ms to 5.4444/3.5720 ms, while routing/input improve from 0.2886/2.1621 ms to 0.1251/2.0325 ms. Activation and combine remain substantial costs. These are profile-stage observations, separate from the end-to-end median speedups.

All 36 profiled calls passed output checks, all source-hash checks passed, and there were no unmatched or unclassified kernels. Sources: [analysis JSON](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final_profile_analysis.json), [stage CSV](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final_profile_breakdown.csv), [call metadata](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final_profile_metadata.json), and [Nsight trace](/workspace/artifacts/thor_moe_mega_int8_20261021/measurement/moe_mega_final.nsys-rep).

Package/source/native/script hashes remained unchanged during measurement. Raw JSON/CSV, logs, validation records, benchmark sources, native binaries and profile captures are kept outside Git in [the artifact bundle](/workspace/artifacts/thor_moe_mega_int8_20261021). Historical performance reports remain unchanged.
