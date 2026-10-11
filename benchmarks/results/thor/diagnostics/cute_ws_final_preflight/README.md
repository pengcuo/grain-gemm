# Independent CuTe candidate preflight

These are private probe checks of the repaired SM110 2SM pipeline, not the final native API validation or the final 19-shape performance report. The probe invokes both pipeline stage counts directly and uses a failing stub for fallback; wrapper and fallback behavior remain for the final full-library tests. No shared library is archived.

The 2SM repair makes each of 256 scale consumers release its own empty-barrier arrival and each of 32 scale producers wait for its async SA/SB copies before publishing its own full-barrier arrival. Invalid SB halves use async zero-fill. Waits and shared scale loads have compiler memory clobbers. Both CTAs join before 2SM TMEM allocation. AB/TMEM pipelines and exact FP32 fmaf/scales layout remain the same.

`ws2_all-racecheck.log` and `ws2_math-racecheck.log` report zero errors. `ws2_all-memcheck-explicit.log` reports zero errors with production `--report-api-errors explicit`; the extended API diagnostic is retained in `ws2_all-memcheck.log` and reports the container TileIR initialization error at cudaGetDeviceCount before the first kernel.

`ws2_results.json` retains every timing sample. Three same-input medians (P2 / CUT32) are 18.05 / 16.65 us for 1024 cubed, 608.43 / 627.10 us for 4096 cubed, and 4532.60 / 4775.93 us for 2048 x 151936 x 1536. These are preflight measurements, not final tuned API results.

The final source snapshot differs from the compiled snapshot only in two explanatory comments; the manifest identifies both SHA256 hashes and reproduction commands. Probe scripts retain their original absolute paths and depend on the workspace reference helpers. Final complete-library sanitizer and performance reports are stored separately.
