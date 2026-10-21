# External source and license provenance

This directory contains benchmark orchestration, an adapter, a narrow host-guard
patch and source verification helpers. It does not vendor DeepInfer/DeepGEMM
device kernels or compiled binaries. The helpers that reproduce the upstream
buffer/API and source-attestation conventions are provided under the upstream
MIT terms where applicable; the complete notices are retained below.

| Dependency | Measured revision | License |
|---|---|---|
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | `8b16149741a4935e1dec8cea27853f273b374c5b` | See its pinned checkout |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), vendored by DeepInfer | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` | [MIT, DeepSeek 2025](LICENSES/DEEPGEMM-LICENSE) |
| [CUTLASS](https://github.com/NVIDIA/cutlass), DeepGEMM submodule | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` | [BSD-3-Clause, NVIDIA](LICENSES/CUTLASS-LICENSE) |
| [deep_jit](https://github.com/deepseek-ai/deep_jit), DeepGEMM submodule | `2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43` | [MIT, DeepSeek 2026](LICENSES/DEEPJIT-LICENSE) |

`compatibility.patch` modifies only four host architecture acceptance conditions
in two DeepGEMM headers. BF16 accepts exact SM110 in addition to original major
10; the inherited FP8 compatibility conditions accept major11. Device source,
weight transformations, scheduling and the real JIT target remain unchanged.
This is experimental compatibility, not an upstream support claim.

The builder copies the pinned upstream sources and actual CUTLASS/deep_jit
headers into a separate user-selected directory, preserving their license files.
The generated source, build log, manifest and library remain outside Git.
Any redistribution of those generated files must preserve the applicable full
notices. The CUTLASS license describes additional terms for its CuTeDSL Python
directory; this experiment uses its external C++ headers.

Cosmos kernels remain external. The benchmark uses the repository's existing
`benchmarks/cosmos_moe_baseline.py` loader, which preserves upstream function
definitions and source hashes. Its provenance and terms are described in
[the repository notices](../../THIRD_PARTY_NOTICES.md).
