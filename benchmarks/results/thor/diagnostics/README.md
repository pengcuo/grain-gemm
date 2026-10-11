# Integration diagnostic: device assertions and register spills

The initial integration was slower than the original Cosmos probe. Its partial
tuning run was stopped after three shapes; it is **incomplete** and is not the
final dispatch or performance comparison.

The first adapter used an explicit `Shape<128,32>` in place of Cosmos cfg20's
`EpilogueTileAuto`. Restoring the structured auto tiler did not improve timing
or spills (`auto_tiler_preflight.json`). Two further micro builds tested the
original scaling expression and explicit scale-first FMA; both produced the
same heavy spills. These experiments ruled out the tiler and FMA expression
as the cause of this performance discrepancy.

The missing option was Cosmos's `-DNDEBUG` release flag. The original probe PTX
contained no `__assertfail` calls; the production SM110a kernels contained 18/34
such calls. Device assertion branches extended live ranges across the pipeline.
The original probe's static SASS had 44 local-load and 43 local-store instructions,
versus 251/314 for the slow adapter's 128x32 variant. Both had the intended
48/216 register-reallocation instructions and 515 FMUL/256 FFMA instructions.
These are static instruction counts, not runtime profiler counts.

A single-configuration micro build with explicit partial-first FMA and `-DNDEBUG`
restored the original probe's PTX virtual-register counts and static SASS counts
(44 local loads / 43 local stores), with zero assertion calls and no FTZ/fast-math.
See `promotion_compilation_comparison.json` and `release_fmaf_probe/analysis.json`.

The production builder now applies `-DNDEBUG` for the Thor CUTLASS target, and
the source requires that option. Public Python validation, native shape/pointer
checks, `can_implement`, and CUDA status checks remain active. The explicit FP32
FMA expression is retained. Final correctness and sanitizer checks use the
release library, followed by complete tuning and an independent API comparison.

The diagnostic JSON and build logs retain the interrupted runs and their source
provenance. See the parent directory for final complete measurements.
