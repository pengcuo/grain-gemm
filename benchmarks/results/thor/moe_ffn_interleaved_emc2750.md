# Thor MoE: physically interleaved gate/up weights

`moe_ffn(..., gate_up_layout="interleaved")` now consumes first-projection weight rows and scale columns ordered `[g0,u0,g1,u1,...]`. Weights keep logical shape `[E,H,2I]` and stride `(H*2I,1,H)`; scales keep contiguous `[E,H/256,2I]`. Down weights/scales are unchanged. The caller prepares this ordering once; no weight/scales repacking runs inside forward. See [API and preparation example](../../../docs/moe_interleaved.md).

GPU frequency was **reduced to nominal 1.2 GHz (actual locked 1197 MHz)**, with **EMC locked at 2750 MHz**. No clock settings were changed. Every trial checked all six min/max/current readings. Maximum sampled temperature was **62.750 C**; cooling begins at 62 C and execution aborts at 70 C.

E=32, H=1536, I=1024 per expert, Top-4, INT8 G256 with FP32 scales, BF16 projection/output rounding, SwiGLU, one Thor GPU. Both layouts use exactly the same input values/scales, expert weight values/scales, routes and scores. Interleaving is a permutation of the first projection output channels, applied to both weight and scale columns outside timing.

## Same-run performance

The primary split control uses the current API and the unchanged original split native kernel, isolating kernel organization from Python API changes. All throughput values below are useful-operation **TOPS**, excluding padding. The Gate/Up timer includes fused SwiGLU and activation quantization, so it is not bare GEMM throughput.

| Tokens | Capacity per expert | Split Gate/Up ms (TOPS) | Interleaved Gate/Up ms (TOPS) | Latency reduction |
|---:|---:|---:|---:|---:|
| 4096 | 640 | 1.766 (58.36) | 1.659 (62.13) | 6.1% |
| 8192 | 1152 | 2.790 (73.90) | 2.588 (79.66) | 7.2% |
| 16384 | 2112 | 5.304 (77.74) | 4.850 (85.02) | 8.6% |
| 32768 | 4288 | 10.656 (77.38) | 9.556 (86.30) | 10.3% |

| Tokens | Split full FFN ms (TOPS) | Interleaved full FFN ms (TOPS) | Latency reduction |
|---:|---:|---:|---:|
| 4096 | 3.222 (47.99) | 3.065 (50.44) | 4.9% |
| 8192 | 5.109 (60.53) | 4.892 (63.21) | 4.2% |
| 16384 | 9.145 (67.63) | 8.672 (71.32) | 5.2% |
| 32768 | 17.452 (70.88) | 16.532 (74.82) | 5.3% |

Gate/Up latency falls by **6.1-10.3%**, and complete FFN latency falls by **4.2-5.3%** against the current split control. This does not eliminate another full gate/up global-memory round trip: the baseline original-token gather kernel already fused Gate/Up, SwiGLU and hidden quantization. Global QH/SH, Down output and final combination remain. No measured DRAM-byte savings are asserted for this layout change.

## Frozen predecessor control

The benchmark also imports an immediately frozen pre-change package, without the new layout keyword. Its full output and intermediate bits match both current layouts. These additional timings are retained as a control; they are not used to inflate the primary comparison above.

| Tokens | Frozen split full FFN ms | Current split full FFN ms | Frozen split Gate/Up ms | Current split Gate/Up ms |
|---:|---:|---:|---:|---:|
| 4096 | 3.205 | 3.222 | 1.770 | 1.766 |
| 8192 | 5.062 | 5.109 | 2.787 | 2.790 |
| 16384 | 9.487 | 9.145 | 5.301 | 5.304 |
| 32768 | 17.511 | 17.452 | 10.500 | 10.656 |

At 16384 tokens, the frozen and current split full-call spans differ more than their fused stage times. Eager CUDA-event spans include CPU capacity synchronization, allocations and launch gaps; the raw round medians are retained. The current-split comparison is the more direct same-API test.

## Kernel change

- True interleaved Linear storage replaces two separate N128 gate/up projections with an M128/N256 two-SM MMA over 128 matching gate/up pairs. Both CTA peers gather original INT8 tokens into shared memory; H1536 reuses cached A across the two I128 halves.
- Adjacent gate/up values occupy the same consumer register fragment. The output loop processes four independent hidden values together, preserving the two BF16 rounding points and FP32 `SiLU(gate) * up` expression. G256 reduction/quantization remains inside this launch.
- Two operand stages, four accumulators and 512 TMEM columns are retained. Shared memory is 232192 bytes. CUDA 13.2 reports CW16: 96 registers, 16-byte stack, 36-byte spill stores / 140-byte spill loads; CW8: 168 registers, no spills. The faster measured version was selected rather than selecting by spill count alone.
- The split CUDA implementation, existing native binaries, single-GEMM code, SM120/GB10 native sources and tuning configurations are unchanged. The default remains `gate_up_layout="split"`; auto uses the matching native build or layout-aware Triton fallback.

## Measurement and validation

Full FFN uses eager public `moe_ffn(auto)` CUDA-event spans, including public validation, routing metadata, the CPU capacity read, allocations, fused Gate/Up/activation/quantization, native Down and combination. Router logits/top-k selection, input/weight quantization, one-time weight permutation, compilation and correctness checks are outside timing. The isolated stage replays the actual public first-projection launch twice per CUDA Graph replay.

One warm full call per method precedes three rounds of two samples; method order rotates/reverses. A shared 256 MB cache flush is outside each timer. Reported times are medians of round medians. Full FFN useful operations are `6*T*4*1536*1024`; Gate/Up useful operations are `4*T*4*1536*1024`, excluding non-GEMM arithmetic. The isolated two-call replay has a common cache policy across methods; it is not two individually cold calls.

- **285 passed, 2 skipped** across the six existing MoE regression files plus the new interleaved tests. Coverage includes BF16/FP16 fallback, masks, signed scales, duplicate/invalid routes, empty experts/input, native pointer identity (no runtime repacking), optional-build hashes/ABI, nondefault streams and changing values/routes on graph replay.
- All four performance shapes pass full QH values, FP32 hidden-scale bits, final BF16 output bits, counts and padding equality against the frozen predecessor and current split implementation, before and after timing. Ten additional native/Triton edge cases and independent two-token CPU references pass.
- Compute Sanitizer memcheck passes CW8 and CW16 over three shapes, with H=256/512/1536, I=256/512/1024, R=1/3/4, partial clusters, empty experts, duplicates, masked NaN scales and graph replay after zeroing counts/scales: **0 errors**.
- All source, native-library, baseline-manifest and benchmark-script fingerprints remain unchanged through the final measurement. The final CUDA SHA256 is `a9820402ae6df9488ae3d8c4536c6452dccf8a23e8d476ae5bea5db2245f9764`; library SHA256 is `8f30881cab495e1a6e0d28200c43e47c9afe92111dcaf2dffc23611ee032311a`.

Build and reproduce:

```bash
python tools/build_cutlass_moe.py --cutlass-dir /workspace/cutlass
python tools/build_moe_gather.py --cutlass-dir /workspace/cutlass
python tools/build_moe_interleaved.py --cutlass-dir /workspace/cutlass
PYTHONPATH=src python benchmarks/bench_moe_interleaved.py \
  --before-package-dir /workspace/artifacts/thor_moe_interleaved_20261021/before_package/grain_gemm_before_interleave \
  --tokens 4096 8192 16384 32768 --rounds 3 --samples 2 \
  --output /tmp/moe_interleaved_reproduction.json
```

Raw samples/CSV, clocks/temperatures, source and native snapshots, candidate screening, build/test/memcheck logs, harnesses and SHA256 manifest are outside Git in `/workspace/artifacts/thor_moe_interleaved_20261021/`. Source uses the existing external CUTLASS dependency at `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
