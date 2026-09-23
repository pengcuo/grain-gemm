# GB10 LLM prefill: Triton and CuTe

[Full M/N/K tables](report.md) · [CSV](results.csv) · [Raw measurements](results.json) · [Experiment scripts](../../../experiments/prefill/README.md)

The completed historical sweep contains **51 unique GEMM shapes / 54 workload uses**, with total prefill tokens 512, 1024, 2048 and 4096. It covers attention linear projections, shared/routed expert projections, router projections and two vocabulary sizes. Routed-expert M assumes balanced top-4 routing over 32 experts. It is not a grouped-MoE or end-to-end model benchmark.

All inputs are prequantized INT8, with G256 FP32 scales and BF16 output. CUDA Graph timing includes group scaling and output conversion and excludes quantization, allocation, compilation and host dispatch. Throughput is **TOPS = 2MNK/time**; historical equivalent-TFLOPS columns use the same operation count.

“Default” refers to the forced-backend configuration recorded at measurement time. These measurements precede the three exact M256 dispatch entries now integrated into the package. Every run already specified BF16 output explicitly, independently of the later API default change. The sweep resumed across host sessions; metadata and all rounds remain in JSON.

The raw JSON and CSV are unchanged. [manifest.json](manifest.json) records their original hashes and identifies the report’s editorial changes. M1024/M2048 shape overlap with later CUTLASS experiments is intentional: these are separate runs and must not be mixed to calculate speedups.
