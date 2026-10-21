# Thor interleaved INT8 Mega experiment

This is an explicitly selected, experimental implementation. It does not change
`grain_gemm.moe_ffn`, `backend="auto"`, the existing Thor fast path, or GB10.
The preserved `v3vq` kernel uses one persistent two-SM cluster per expert M128
tile for interleaved Gate/Up, SwiGLU, G256 quantization and Down. Quantized hidden
activations and scales stay in shared memory. The Down epilogue reuses 32 KiB of
dead Gate/Up input scratch, then each warp copies a contiguous BF16 output row
with aligned `uint4` stores. BF16 expert outputs still go to global memory for
the existing separate weighted combine kernel; final combine is not fused.

This experiment remains outside automatic dispatch. See the same-run
[performance report](../../benchmarks/results/thor/moe_ffn_interleaved_mega_emc2750.md)
for measured latency, throughput and the DeepInfer FP8 reference. Fusion alone
is not a claim that this implementation is faster.

The core derives from the frozen `v3vq` source with original SHA256
`02313f5c990a1b1727edae5b8082f3717d15d5fe0917a65b6715b102b02c5570`;
the changes are the coalesced Down writeback from the measured
`v3vqlut_coalesced` candidate and the optional exact SiLU table. The writeback
staging adds no shared-memory allocation and preserves BF16 output rounding.
It preserves NVIDIA's full BSD-3-Clause source notice and uses external CUTLASS
CuTe primitives. Scheduling is inspired by DeepInfer; no DeepInfer/DeepGEMM
kernel source is copied. See [third-party notices](../../THIRD_PARTY_NOTICES.md).

## Build

Build the measured table variant at the explicit API's default path, from the
repository root, using CUDA 13.2 and the existing pinned CUTLASS:

```bash
python experiments/thor_interleaved_mega/build.py \
  --cutlass-dir /path/to/cutlass --silu-lut \
  --output-dir experiments/thor_interleaved_mega/build
```

The builder requires clean tracked CUTLASS headers at commit
`098de2a652cf8f00fd70b2df54051c7eccbb855a`, accepts only `sm_110a`, and never
executes a GPU kernel. `--nvcc`, `--output-dir` and `--dry-run` are available.
Libraries and source/compiler/hash metadata are written under the ignored
`experiments/thor_interleaved_mega/build/`, separate from production binaries.
For an isolated table build instead, use:

```bash
python experiments/thor_interleaved_mega/build.py \
  --cutlass-dir /path/to/cutlass --silu-lut
```

This writes `build/lut/libgrain_moe_mega.so`, leaving the default build intact.
Pass that path as `library=` to the explicit API or `--library` to validation.
For the non-table control, omit `--silu-lut` and set `--output-dir` to
`experiments/thor_interleaved_mega/build/plain`.
The loader rejects missing, stale or incompatible metadata and library bytes
on first load. Library handles and metadata are cached per process; **restart
the Python process after rebuilding the same library path**.

## Explicit API

```python
from experiments.thor_interleaved_mega.local_ctypes_api import moe_ffn_mega

y = moe_ffn_mega(
    qx, w_gate_up, w_down, topk_weights, topk_ids,
    scale_x=scale_x,
    scale_gate_up=scale_gate_up,
    scale_down=scale_down,
    gate_up_layout="interleaved",
    consumer_warps=16,  # Also supports 8.
)
```

Tensor arguments follow the public MoE API:

| Argument | Shape / type / storage |
|---|---|
| `qx` | INT8 `[T,H]`, contiguous, including supplied -128 values |
| `scale_x` | FP32 `[T,H/256]`, contiguous |
| `w_gate_up` | INT8 `[E,H,2I]`, stride `(H*2I,1,H)` |
| `scale_gate_up` | FP32 `[E,H/256,2I]`, contiguous |
| `w_down` | INT8 `[E,I,H]`, stride `(I*H,1,I)` |
| `scale_down` | FP32 `[E,I/256,H]`, contiguous |
| `topk_ids` | INT32/INT64 `[T,R]`, contiguous |
| `topk_weights` | FP32 `[T,R]`, contiguous, used without renormalization |
| `token_mask` | Optional contiguous BOOL `[T]`; false rows return zero |
| Result | BF16 `[T,H]` |

Gate/Up weight output columns **and scale columns** must be supplied as
`[g0,u0,g1,u1,...]`. No runtime weight permutation occurs. See the
[preparation example](../../docs/moe_interleaved.md). Down storage is unchanged.
The current prototype is limited to **H=1536, I=1024, G=256, BF16 output, SM110,
eager execution**, and 16-byte-aligned input/weight/scale pointers. Empty input,
empty experts, masked invalid IDs and duplicate expert slots follow the public
semantics; unmasked invalid IDs raise. `backend="auto"` and `"cutlass"` both
explicitly select this prototype; other backends are rejected. CUDA Graph capture
is rejected because expert capacity requires an eager host statistics read.

Projection results use the original BF16 rounding points. SwiGLU is FP32. The
plain build evaluates the original `expf` and correctly rounded division; the
selected table build looks up the exact same expression for the BF16 gate.
G256 quantization uses a
correctly rounded reciprocal plus multiplication except within a guarded region
around half-integer boundaries, where it falls back to the original division.
The guard preserves the quantized integer result and does not change scales.
See the [rounding proof](quantization_proof.md).
Scores multiply the BF16 Down results before the original FP32 combination.

`debug_moe_ffn_mega` accepts the same arguments and returns the actual `output`,
`qh`, `sh`, `expert_y`, `counts` and `capacity` produced by an untimed debug
launch. Debug QH/SH are extra global writes; they are absent from normal timing.
`candidate.py:make_experiment(context)` is the thin comparison-harness adapter.
It regenerates validation/routing/capacity/allocations on every timed call.

At reduced GPU 1.2 GHz (actual 1197 MHz) and EMC 2750 MHz, the measured table
build reduced complete-call latency by 2.1%, 9.0%, 2.5% and 2.6% at
4096/8192/16384/32768 tokens. It remains slower than DeepInfer FP8. The
[performance report](../../benchmarks/results/thor/moe_ffn_interleaved_mega_emc2750.md)
includes the paired results, useful TOPS, separate profile and rejected output
fusion trials. These measurements apply to E32/H1536/I1024/Top4; they do not
establish a ranking for other shapes.

## Correctness and measurement

```bash
PYTHONPATH=src python experiments/thor_interleaved_mega/validate.py \
  --output /tmp/thor_interleaved_mega_validation.json
```

Validation is restricted to Thor and checks the current public native interleaved
path against the experiment for signed/zero scales, INT8 -128 inputs, duplicate
routes, masks, empty input, empty experts, skewed routing, both consumer-warp
choices and a nondefault stream. It compares actual full QH/SH and useful BF16
expert-output/output bits. Only the process running this script should own GPU
execution while measuring. The script reads all six clock settings and requires
GPU1197MHz and EMC2750MHz, cools at62C to54C, and aborts at70C. It never sets clocks.

A full-FFN performance report must compare the current interleaved public path,
this experiment and DeepInfer FP8 in the same run at reduced nominal GPU1.2GHz
(actual1197MHz), EMC2750MHz. Use useful `6*T*R*H*I` projection operations for
INT8 TOPS / FP8 TFLOPS, including both Gate/Up and Down. Input/weight preparation
is outside timing. Public eager INT8 includes validation, routing, CPU capacity
reads and per-call allocation; FP8 reuses its prepared SymmBuffer. State that
workspace/API distinction alongside results, and retain the different G256 INT8
versus G32 UE8M0 FP8 numerical contracts. No pure-format or equal-quality claim
follows from the full-call comparison. Keep raw JSON/CSV, logs, binaries, traces
and frozen measured sources in the external artifact bundle; this directory
tracks source and documentation.

## Optional exact SiLU table contract

The optional `--silu-lut` build exports `grain_moe_mega_cluster_initialize(stream)`. It initializes an
immutable per-device table once, records a completion event and makes another
stream wait before use; a Python lock also protects concurrent first use.

A BF16-rounded gate has only 65536 bit patterns. A FP32 table generated on the same
GPU/compiler using the identical expression `g/(1+expf(-g))` can therefore store
every possible gate result exactly. Indexing the same BF16 gate pattern and
multiplying by the unchanged Up value preserves each FP32 hidden value, then
amax, scales and quantized values. Exhaustive GPU testing over all 65536 patterns
must check the actual table against the original expression, including zeros,
subnormals, infinities and NaN behavior, before accepting such a variant. A
256 KiB table trades repeated math for indexed cache reads; performance still
requires measurement. Validation checks initialization on one stream followed by
use on a second stream with the cached event dependency, plus all 65536 table
entries. This is a targeted stream-order check, not a GPU concurrency stress test.

The portable two-way full-FFN benchmark compares the public interleaved INT8
path with this explicit experiment using identical prequantized inputs, weights,
scales and routes:

```bash
PYTHONPATH=src python experiments/thor_interleaved_mega/benchmark.py \
  --tokens 4096 8192 16384 32768 --rounds 3 --samples 2 \
  --output /tmp/thor_interleaved_mega_benchmark.json
```

Use `--tokens 4096 --smoke-only` for exact final-output checks without timed
repetitions; `--library` selects a separately built experimental library. JSON
and CSV include eager full-call latency, useful INT8 TOPS, speedup, alternating
sample order, all clock/temperature readings, and source/library fingerprints.
Input quantization and weight interleaving are outside timing. Both complete
API calls retain validation, routing, host capacity statistics and allocations;
CUDA Graphs are not used. This portable comparison contains no FP8 baseline.
