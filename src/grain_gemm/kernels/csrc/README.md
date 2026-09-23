# Native source layout

Group native implementations by their target architecture family, and name
translation units by operation and backend. The Python API and backend names
remain independent of the source layout.

| Directory | Current scope |
| --- | --- |
| `sm12x/` | Direct CuTe INT8 G256 implementation for SM121 and experimental SM120 builds |
| `sm12x/cutlass/` | CUTLASS composition for the same operation and targets |

GB10 is the measured and tuned device for the current native kernels. The SM120
build path is opt-in and has not been validated on RTX hardware. The `sm12x`
directory is not a promise of support for every future SM12x device.

Future H100 and Thor / T5000 implementations should be added under `sm90/` and
`sm110/`, respectively, when they exist. Shared implementation details can move
into a common directory when there is an actual second user. Keep internal
symbols distinct when linking several implementations through a shared public
launcher.

Tuning data live separately under `../configs/`. Names include the measured GPU,
SM target and group size, such as `gb10_sm121_g256.json` and
`gb10_sm121_g256_cutlass.json`. GPUs with the same SM version can still need
different tuning choices.

Register new sources in the builders, source-hash checks and package-data rules.
Historical experiment snapshots and their recorded source paths stay unchanged.
