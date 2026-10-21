# Thor: padding-aware INT8 MoE versus DeepInfer FP8 Mega, fresh retest

Retested after the per-expert padding-skip update. GPU frequency was **reduced
to nominal 1.2 GHz (actual locked 1197 MHz)**; EMC was locked to **2750 MHz**.
All GPU/EMC minimum, maximum and current frequency readings were verified during
the run. No clock settings were changed. The highest sampled GPU temperature
was **62.406°C**; cooling begins at 62°C and execution aborts at 70°C.

E=32, H=1536, I=1024, Top-4, no shared experts, one Thor GPU, no multi-GPU
communication. Original BF16 inputs/weights, expert IDs and routing scores are
shared by all methods; scores are FP32 rounded through BF16. The first four
rows use random distinct Top-4 expert IDs per token. The last row has 16 experts
with exactly 4096 assignments each and 16 with 1000 each, at capacity 4096.

| Input tokens | INT8 before padding skip, ms | Current INT8 ms (TOPS) | DeepInfer FP8 Mega ms (TFLOPS) | INT8 / FP8 time |
|---:|---:|---:|---:|---:|
| 4096 | 3.116 | **3.098 (49.9)** | 2.327 (66.4) | 1.331× |
| 8192 | 5.198 | **5.160 (59.9)** | 4.095 (75.5) | 1.260× |
| 16384 | 9.664 | **9.650 (64.1)** | 7.401 (83.6) | 1.304× |
| 32768 | 18.659 | **18.706 (66.1)** | 13.982 (88.5) | 1.338× |
| 20384 | 16.144 | **13.035 (59.0)** | 9.037 (85.1) | 1.442× |

With random routing, current INT8 takes **1.26–1.34×** DeepInfer FP8's time,
consistent with the preceding comparison. There are few wholly padded clusters
to remove in these populations. The uneven case improves from **16.144 ms to
13.035 ms**, a **19.26%** reduction. Against FP8's **9.037 ms**, the INT8/FP8 time
ratio decreases from **1.786× to 1.442×**. Padding skipping helps this case but
does not close the end-to-end gap.

## Timing scope and protocol

These are complete eager FFN CUDA-event spans. Current INT8 calls the actual
public `grain_gemm.moe_ffn(..., backend="auto")`, including validation,
routing/counting and its CPU capacity read, runtime allocation, input
quantization/packing, selected gate/up plus SwiGLU/hidden quantization, down
projection and final weighted combination. The before package was frozen
immediately before the padding-skip change and already has cached two-SM fused
gate/up at 4096/8192/16384 tokens. At 32768 and 20384, both INT8 packages use
separate gate/up and activation; their counted native configuration choices
are recorded with the samples.

FP8 includes BF16-to-FP8 input quantization directly into the Mega input buffer,
ID/score copies, output allocation and the Mega kernel's dispatch, gate/up,
activation/quantization, down and combination. Router logits/top-k, compilation,
one-time weight quantization/layout transforms and reusable FP8 Mega workspace
setup are excluded. No full-call CUDA Graph timing is used.

Each method receives two short warm calls, followed by three rounds of three
samples. Method order rotates and reverses so all three implementations occupy
each position equally. A common preallocated 256 MB L2 flush runs outside each
timing interval. The table reports the median of round medians. Throughput is
`6 * T * 4 * 1536 * 1024 / time`, counting useful projection operations and
excluding padding and non-GEMM arithmetic. INT8 uses TOPS; FP8 uses TFLOPS.

## Correctness and reference implementation

All five sizes passed full before/after INT8 QH bits, intermediate FP32 scale
bits, expert counts, zero padding and final BF16 output bits. Repeated complete
outputs were bitwise stable before and after timing. The direct FP8 input
producer matched the upstream cast's quantized values and packed scales
bitwise. FP8's first-call expert counts matched routing counts. Both methods
passed checks against independent CPU references using their respective
quantized arithmetic; a common unquantized FP32 reference is also recorded.

DeepInfer: `8b16149741a4935e1dec8cea27853f273b374c5b`;
vendored DeepGEMM: `1fc9c66975ee322d628b3288534ca0e9ce12a64d`.
The existing SM110 compatibility build extends three host architecture checks;
its device arithmetic, layouts and scheduling remain unchanged. The one-rank
buffer path uses no NCCL or expert-parallel communication.

FP8 uses E4M3 G32 UE8M0 scales, `fast_math=True`, and applies routing scores
before intermediate quantization/down. INT8 uses G256 FP32 scales, the public
GrainGEMM rounding contract, and scores after BF16-rounded down outputs. This
is a performance comparison of different numerical contracts; it does not
establish equal model accuracy. Source, payload, script and native-binary
fingerprints were unchanged throughout the run.

Raw JSON/CSV samples, clocks/temperatures, checks, provenance, harnesses and
frozen INT8 packages are outside Git at
`/workspace/artifacts/thor_moe_padding_deepinfer_retest_20261021/`.

Related: [padding-skip implementation and isolated projections](moe_ffn_padding_skip_emc2750.md),
[preceding cached two-SM comparison](moe_ffn_cached_2sm_emc2750.md).
