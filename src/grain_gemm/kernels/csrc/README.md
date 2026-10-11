# Native source layout

Group native implementations by their target architecture family, and name
translation units by operation and backend. The Python API and backend names
remain independent of the source layout.

| Directory | Current scope |
| --- | --- |
| `sm110/` | Thor SM110 independent CuTe TMA/UMMA pipelines and CUTLASS G256 collective, with shared warp MMA candidates |
| `sm12x/` | Direct CuTe INT8 G256 implementation for SM121 and experimental SM120 builds |
| `sm12x/cutlass/` | CUTLASS composition for the same operation and targets |

GB10 and Thor have separate native measurements and tuning tables. The SM120
build path is opt-in and has not been validated on RTX hardware. The `sm12x`
directory is not a promise of support for every future SM12x device.

Future H100 implementations should be added under `sm90/` when they exist.
Thor / T5000 is implemented under `sm110/`. The Thor translation unit includes
the existing warp MMA implementation as additional tuning candidates. Its
CuTe IDs 26–29 instantiate independent warp-specialized pipelines from
`int8_g256_pipeline.cuh` and `int8_g256_pipeline_1sm.cuh`: two-SM 256×256 tiles
with three/two operand stages, and one-SM 128×128 / 128×256 tiles with
three/two stages. Each CTA uses four producer and eight consumer warps,
TMA/UMMA, two TMEM buffers and three prefetched promotion fragments.
FP32 `SA[M,K/256]` is K-group-contiguous and `SB[K/256,N]` is N-contiguous;
both are read directly. INT32 group partials update FP32 totals with `fmaf`
before a single BF16 conversion. Two-SM configs 26/27 support
M multiples of 256 and N multiples of 128, including a final N128 half-tile.
TMA and shared scales zero the invalid weight columns, output stores skip them,
and all threads still participate in barriers. One-SM configs 28/29 require
full BM/BN tiles. Other tile/alignment limitations use warp-MMA config 0.
Fresh descriptors and the caller's current stream preserve Graph
input updates without cached pointers.

Unmeasured Thor explicit CUDA calls start with config 27 for M/N multiples of
256, config 28 for M/N multiples of 128, and config 17 otherwise. Unmeasured
auto calls retain Triton; SM120/GB10 defaults remain unchanged.

The SM110 CUDA source hash covers four files: `sm110/int8_g256_cute.cu`, both
pipeline `.cuh` headers and `sm12x/int8_g256_cute.cu`. Package-data rules
include the new `.cuh` headers. The independent CuTe kernels apply the same
pipeline ideas as CUTLASS IDs 32/33 without calling that backend.
SM120/GB10 sources, configuration IDs and build flags remain unchanged.
The SM110 CUTLASS adapter adds a two-SM G256 pipeline
and uses architecture-specific shadow headers, while preserving its 32 shared
warp MMA candidates. Its source hash covers all transitive project dependencies.
Shared implementation details can move
into a common directory when there is an actual second user. Keep internal
symbols distinct when linking several implementations through a shared public
launcher.

Tuning data live separately under `../configs/`. Names include the measured GPU,
SM target and group size, such as `gb10_sm121_g256.json` and
`gb10_sm121_g256_cutlass.json`. GPUs with the same SM version can still need
different tuning choices.

Register new sources in the builders, source-hash checks and package-data rules.
Historical experiment snapshots and their recorded source paths stay unchanged.
