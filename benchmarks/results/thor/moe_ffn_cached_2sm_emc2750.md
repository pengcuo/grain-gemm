**Thor INT8 MoE FFN: cached two-SM gate/up retry — 2026-10-21**

This retry reduces complete public FFN latency by **3.9%, 3.5% and 4.4%** at 4096, 8192 and 16384 tokens. The 32768 path is unchanged and measures effectively flat. **DeepInfer FP8 remains 1.26–1.33× faster; this implementation has not reached parity.**

GPU frequency was **reduced to nominal 1.2 GHz, actual 1197 MHz**. **EMC was locked at 2750 MHz**. Every recorded minimum, maximum and current reading matched both locks. Configuration: 32 experts, H=1536, I=1024, Top-4, no shared experts; identical original BF16 inputs/weights, routes and FP32 scores rounded through BF16.

| Tokens | Before this retry: INT8 ms (TOPS) | Updated INT8 ms (TOPS) | Latency reduction | DeepInfer FP8 ms (TFLOPS) | INT8 / FP8 time |
|---:|---:|---:|---:|---:|---:|
| 4096 | 3.219 (48.0) | **3.093 (50.0)** | 3.9% | 2.339 (66.1) | 1.32× |
| 8192 | 5.348 (57.8) | **5.161 (59.9)** | 3.5% | 4.086 (75.7) | 1.26× |
| 16384 | 10.088 (61.3) | **9.642 (64.1)** | 4.4% | 7.418 (83.4) | 1.30× |
| 32768 | 18.782 (65.9) | **18.804 (65.8)** | ≈0% | 14.146 (87.4) | 1.33× |

Both INT8 columns call the actual public `moe_ffn(..., backend="auto")`. The baseline is the frozen **immediately-before-this-retry working package**, including the previous one-SM fusion at 4096/8192; it is not a pristine Git checkout. The updated package selects the cached two-SM kernel at 4096/8192/16384 and retains native separate G1/activation at 32768. Compare these contemporaneous columns; timings in older reports were separate runs.

Raw complete measurements: [JSON](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final.json), [CSV](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final.csv), [log](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final.log). The [previous report](moe_ffn_mega_int8_emc2750.md) remains unchanged.

Each method has two warm calls followed by three rounds of three samples, with balanced rotating/reversing order and a common 256 MB L2 flush outside each timer. Latency is the median of round medians. CUDA events span the full eager call, including validation, routing and host capacity synchronization, input quantization/packing, both projections, activation, output combine and runtime allocation. **No full-call CUDA Graph timing is used.** Router logits/top-k, one-time weight quantization, compilation and reusable FP8 Mega workspace setup are excluded. FP8 includes BF16-to-FP8 input quantization, ID/score copies and output allocation. No multi-GPU communication is used.

Throughput counts useful projection operations only: G1 `4*T*R*H*I`, G2 `2*T*R*H*I`, complete FFN `6*T*R*H*I`. Padding and non-GEMM arithmetic are excluded. INT8 rates are TOPS and FP8 rates are TFLOPS; complete-FFN throughput is distinct from bare GEMM throughput.

The retained kernel uses a two-SM cluster, paired gate/up MMA with a shared input tile, and four paired TMEM accumulator buffers. At H1536 the two I128 halves reuse six cached A blocks. BF16 rounding of both projections still precedes FP32 SwiGLU; a complete G256 hidden group stays FP32 in shared memory before INT8 quantization. The CUDA compiler reports 94 registers per thread and zero spills for the selected 16-consumer-warp variant. It uses 232192 bytes of shared memory against this device's 232448-byte opt-in limit.

The public signature, mathematical transposed-Linear weight views, FP32 scale layouts, quantization/rounding order, masks and weighted reduction are unchanged. SM120/GB10 and single-GEMM code are unchanged. Routing, input packing, G2 and combine remain separate stages. The original native G1/G2 library is byte-identical to the baseline.

The next table uses **three separately profiled calls per method/size**, showing additive **mean exclusive GPU time**, rather than the uninstrumented medians above. G1 includes SwiGLU and activation quantization in both columns.

| Tokens | Before: G1 + activation/quant ms | Updated: G1 + activation/quant ms (useful G1 TOPS) | Stage latency reduction |
|---:|---:|---:|---:|
| 4096 | 1.7644 | 1.5890 (64.9) | 9.9% |
| 8192 | 2.7050 | 2.5323 (81.4) | 6.4% |
| 16384 | 5.2145 | 4.7970 (86.0) | 8.0% |
| 32768 | 9.5089 | 9.4868 (86.9) | ≈0% |

**Updated INT8 stage breakdown: mean ms / share (projection TOPS)**

| Stage | 4096 tokens | 8192 tokens | 16384 tokens | 32768 tokens |
|---|---:|---:|---:|---:|
| Routing/counting | 0.0215 / 0.6% | 0.0346 / 0.6% | 0.0646 / 0.7% | 0.1236 / 0.7% |
| Input quantization + packing | 0.3004 / 8.9% | 0.5476 / 10.1% | 1.0852 / 11.0% | 2.0397 / 11.0% |
| Separate gate/up GEMM | — | — | — | 5.3907 / 29.1% (153.0) |
| Separate SwiGLU + quantization | — | — | — | 4.0961 / 22.1% |
| Fused gate/up + SwiGLU + quantization | 1.5890 / 46.9% (64.9) | 2.5323 / 46.6% (81.4) | 4.7970 / 48.7% (86.0) | — |
| Down GEMM | 0.7365 / 21.7% (70.0) | 1.1965 / 22.0% (86.2) | 1.9472 / 19.8% (105.9) | 3.5618 / 19.2% (115.8) |
| Output combine | 0.3428 / 10.1% | 0.7426 / 13.7% | 1.4929 / 15.2% | 2.9879 / 16.1% |
| Routing statistics D2H | 0.0024 / 0.1% | 0.0014 / 0.0% | 0.0015 / 0.0% | 0.0015 / 0.0% |
| Outside GPU activities | 0.3970 / 11.7% | 0.3772 / 6.9% | 0.4516 / 4.6% | 0.3484 / 1.9% |
| Full FFN, profiled | 3.3896 / 100.0% (45.6) | 5.4322 / 100.0% (56.9) | 9.8400 / 100.0% (62.9) | 18.5497 / 100.0% (66.7) |

Stage means plus the outside-GPU residual add to each profiled mean total; no cross-stage overlap was observed. The D2H row counts memcpy activities, excluding their full host-wait cost. Outside-GPU time is event span minus the union of GPU activities, including submission/synchronization gaps and event edges; it is not a direct CPU-time measurement. DeepInfer Mega remains an inseparable kernel in this trace; no fictitious internal G1/G2 split is assigned.

Profile evidence: [analysis](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final_profile_analysis.json), [stage CSV](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final_profile_breakdown.csv), [Nsight Systems trace](/workspace/artifacts/thor_moe_retry2_20261021/measurement/moe_retry2_final_profile.nsys-rep).

Several complete FFN alternatives were actually compiled and tested. Ready-queue G1/G2 with an 8–256-slot activation ring, parallel G2 tasks, and GPU-only pooled/ragged input all produced matching valid outputs in their screened cases but were slower, so they are excluded from production. For example, at 32768 tokens the queue/ring compute measured 19.36 ms versus 13.13 ms for native G1 + activation + G2. Pooled input through final combine measured 25.19 ms versus 18.32 ms for the native stage sequence. These isolated device-stage measurements use short CUDA Graphs and are not the full public eager rates in the main table.

The generic queue engine used a shallower operand/scale pipeline (2 versus production G2 config44's 3 stages), fewer accumulator buffers (2 versus 4), narrower TMEM loads with more waits, and spilling. Its M64/N256 G2 tile requested approximately 25% more logical operand bytes per useful output than config44's M128/N128 tile in a full-tile comparison. These are source-derived request counts, not measured DRAM traffic or a claim about the INT8 hardware ceiling. The evidence shows that fusion must retain a strong compute pipeline; merely fusing the existing stages did not meet the performance goal.

Standalone CUDA/TMA output combine variants matched the existing arithmetic but did not outperform it. Hiding the capacity-statistics copy gave only small, size-dependent gains with speculative padding and was also left experimental. Rejected candidates and their raw checks/timings are retained under the artifact bundle's `screening/` directory.

Correctness: all four formal sizes passed complete QH INT8 bits, FP32 intermediate scale bits, padding, expert counts and final BF16 output comparisons against the frozen baseline, including repeated checks after timing. Independent CPU quantized reference outputs were exact for the checked tokens. Related API, mask, duplicate/invalid-route, boundary, graph-replay, stream and fallback regression tests passed **188 tests with 2 skips**. Compute Sanitizer passed 10 edge cases twice for **each** CW8/CW16 variant with **0 errors**, exercising physical CTA count 3 rounded up to a two-CTA cluster grid.

Validation: [pytest log](/workspace/artifacts/thor_moe_retry2_20261021/verification/retry2_pytest.log), [CW16 memcheck](/workspace/artifacts/thor_moe_retry2_20261021/verification/retry2_installed_memcheck.log), [CW8 memcheck](/workspace/artifacts/thor_moe_retry2_20261021/verification/retry2_installed_memcheck_cw8.log). The maximum sampled temperatures were 63.125°C in the uninstrumented pass and 62.812°C in the profile pass; cooling starts at 62°C and execution aborts at 70°C.

DeepInfer reference: DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`, vendored DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`, with three host architecture-acceptance guard changes for SM110 and a separately bit-verified direct input quantizer. Device arithmetic/layout/scheduling are unchanged; this is an SM110 compatibility run. FP8 is E4M3 G32 UE8M0, `fast_math=True`, with scores applied before intermediate quantization/G2. INT8 uses G256 FP32 scales and scores after BF16-rounded G2. This compares performance across different numerical contracts, without an equal-accuracy or model-quality claim.

Source/build hashes, native binaries, the frozen baseline, reference payload, harnesses and raw screening results are preserved in the [artifact bundle](/workspace/artifacts/thor_moe_retry2_20261021/README.md). Only readable reports and source belong in Git; logs, JSON measurements and binaries remain outside the repository.
