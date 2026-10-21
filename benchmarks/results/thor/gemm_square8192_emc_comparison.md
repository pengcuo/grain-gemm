# Thor single 8192-cubed GEMM: CuTe INT8 versus default Torch BF16

Measured 2026-10-21 20:07 CST. **M=N=K=8192**, GPU locked at **1197 MHz (reduced to approximately 1.2 GHz)**. EMC was tested at locked 2750 and 4266 MHz. After testing, EMC was restored to dynamic 665.6–4266 MHz; GPU limits remained unchanged.

Unlike the preceding small rectangular shapes, this square GEMM exceeds **200 INT8 TOPS**: **211.24 TOPS** at EMC2750 and **220.99 TOPS** at EMC4266. The default Torch BF16 measurements reach **75.43** and **116.38 TFLOPS**, respectively. These are measurements of the named implementations, not theoretical hardware ceilings or an exhaustive search for the best BF16 algorithm.

## Repeated-input results

| EMC MHz | CuTe INT8 ms | INT8 TOPS | Torch BF16 ms | BF16 TFLOPS | BF16 / INT8 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2750 | 5.2052 | 211.24 | 14.5757 | 75.43 | 2.80× |
| 4266 | 4.9754 | 220.99 | 9.4473 | 116.38 | 1.90× |

INT8 uses G256 operands and contiguous FP32 scales, with BF16 output. CuTe uses the independent CUDA **config #26**, selected by the repository’s existing 8192-square dispatch entry. This differs from fixed #27 in the preceding rectangular experiments; no new tuning is performed here. BF16 is default `torch.mm(x, weight.T, out=out)` with BF16 inputs/output. Both use the same original BF16 data, quantized for INT8 before timing.

## Cold-cache results

| EMC MHz | CuTe INT8 ms (TOPS) | Torch BF16 ms (TFLOPS) |
| ---: | ---: | ---: |
| 2750 | 5.3009 (207.42) | 14.5783 (75.42) |
| 4266 | 5.0341 (218.41) | 9.5116 (115.60) |

## Method and validation

A is contiguous `[8192,8192]`; B is a transpose view of contiguous Linear weights of the same shape. INT8 scales are contiguous activation `[8192,32]` and weight `[32,8192]`. Input generation, quantization, graph capture, host dispatch, routing and other MoE operations are excluded. Output buffers are preallocated.

Each CUDA graph replay contains one GEMM. Each method/cache mode uses five rounds of five samples, reporting the median of round medians. Cold samples clear a 256 MB buffer outside the timed events and alternate method order. Warm samples execute each method in its own repeated-input block without a cache flush. There is no sustained warmup. Policies are measured sequentially. Throughput uses dense `2*M*N*K / time`; INT8 is TOPS and BF16 is TFLOPS.

Both runs passed sampled bit-exact INT8 BF16-output comparisons against independent CPU INT32 G256 dot products with FP32 scaled accumulation. BF16 outputs passed sampled CPU FP64-reference comparisons (`rtol=1/128`, `atol=0.05`). Each numerical comparison samples 27 elements across three rows and nine columns, before and after timing. Every output element was finite. These are sampled numerical comparisons, not exhaustive comparisons.

All six GPU/EMC min/max/current readings match the requested locks before and after samples. Source/native fingerprints were stable during both runs and identical across runs. No production kernels or dispatch entries were modified.

| EMC MHz | Maximum observed GPU temperature |
| ---: | ---: |
| 2750 | 63.906 °C |
| 4266 | 66.218 °C |

The guard pauses at 62 °C until 54 °C and aborts at 70 °C. The benchmark only reads clock settings.

## Reproduction

Set the matching EMC lock on the host while retaining the GPU1197 MHz lock, then run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --rows 8192 --n 8192 --k 8192 --cute-config 26 \
  --emc-hz 2750000000 --output /tmp/thor_gemm_square8192_emc2750.json

PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/bench_gemm_emc.py \
  --rows 8192 --n 8192 --k 8192 --cute-config 26 \
  --emc-hz 4266000000 --output /tmp/thor_gemm_square8192_emc4266.json
```

Raw JSON/logs and measured source/native snapshots are stored outside Git in `/workspace/artifacts/thor_gemm_square8192_emc_20261021/`.
