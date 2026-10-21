# Thor single 4096-cubed GEMM: CuTe INT8 versus default Torch BF16

Measured 2026-10-21 20:09 CST. **M=N=K=4096**, GPU locked at **1197 MHz (reduced to approximately 1.2 GHz)**. EMC was locked at 2750 and 4266 MHz in sequential runs. After testing, EMC was restored to dynamic **665.6–4266 MHz**, with GPU limits unchanged.

CuTe uses independent CUDA **config #26**, selected by the existing 4096-square dispatch entry. No new tuning was performed. The same script and timing protocol were used for the [8192-square comparison](gemm_square8192_emc_comparison.md).

## Repeated-input results

| EMC MHz | CuTe INT8 ms | INT8 TOPS | Torch BF16 ms | BF16 TFLOPS | BF16 / INT8 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2750 | 0.801248 | 171.53 | 1.207968 | 113.78 | 1.51× |
| 4266 | 0.724800 | 189.62 | 0.917536 | 149.79 | 1.27× |

## Cold-cache results

| EMC MHz | CuTe INT8 ms (TOPS) | Torch BF16 ms (TFLOPS) |
| ---: | ---: | ---: |
| 2750 | 0.850272 (161.64) | 1.375040 (99.95) |
| 4266 | 0.749216 (183.44) | 1.018336 (134.96) |

## Scope, method and validation

INT8 uses G256 quantized operands with contiguous FP32 scales and BF16 output. BF16 uses default `torch.mm(x, weight.T, out=out)`, without changing BLAS preference or precision flags. A is contiguous `[4096,4096]`; B is the transpose of contiguous Linear weights of the same shape. INT8 scales are contiguous `[4096,16]` and `[16,4096]`. INT8 operands are quantized from the same original BF16 input/weight tensors before timing. Outputs are preallocated. Input generation, quantization, graph capture, host dispatch and MoE operations are excluded.

Each graph replay contains one GEMM. Each method/cache mode uses five rounds of five samples; latency is the median of round medians. Cold samples clear a 256 MB buffer outside the timed event and alternate method order. Warm samples execute each method in its own repeated-input block without a cache flush. There is no sustained warmup. Throughput is dense `2*M*N*K / time`, in TOPS for INT8 and TFLOPS for BF16.

Both runs passed sampled bit-exact INT8 BF16-output comparisons with independent CPU INT32 per-G256 dot products and FP32 scaled accumulation. BF16 passed sampled comparisons with independent CPU FP64 references (`rtol=1/128`, `atol=0.05`). Each comparison checks 27 elements across three rows and nine columns before and after timing. All output elements were finite. These are sampled numerical comparisons, not exhaustive comparisons.

All six GPU/EMC min/max/current readings matched their requested locks before and after timed samples. Benchmark/native fingerprints remained stable during both runs and match across runs. No production kernel or dispatch changes were made.

| EMC MHz | Maximum GPU temperature |
| ---: | ---: |
| 2750 | 62.125 °C |
| 4266 | 62.75 °C |

The guard pauses at 62 °C until 54 °C and aborts at 70 °C. The benchmark only reads clock settings.

## Reproduction

Set the matching EMC lock on the host while retaining GPU1197 MHz, then run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --rows 4096 --n 4096 --k 4096 --cute-config 26 \
  --emc-hz 2750000000 --output /tmp/thor_gemm_square4096_emc2750.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --rows 4096 --n 4096 --k 4096 --cute-config 26 \
  --emc-hz 4266000000 --output /tmp/thor_gemm_square4096_emc4266.json
```

Raw JSON/logs and measured source/native snapshots are archived outside Git in `/workspace/artifacts/thor_gemm_square4096_emc_20261021/`.
