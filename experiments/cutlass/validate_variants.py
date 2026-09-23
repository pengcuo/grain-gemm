#!/usr/bin/env python3
"""Check complete outputs, INT8 endpoints, signed/zero scales and pipeline wrap."""

import argparse
from pathlib import Path

from build import VARIANTS
from paired_benchmark import Native

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        parser.error(f"Unknown variants: {', '.join(sorted(unknown))}")
    import torch
    from correctness_fixture import case

    libraries = {name: Native(args.build_dir.expanduser().resolve() / name / "libgrain_cutlass.so",
                              "grain_cutlass_g256") for name in args.variants}
    checked = rejected = 0
    for m, n, k in ((384, 256, 256), (256, 384, 768),
                    (384, 256, 1280), (384, 256, 4352)):
        a, b, sa, sb, expected = case(m, n, k)
        out = torch.empty_like(expected)
        for name, library in libraries.items():
            for config in (11, 15, 23):
                if name == "ungrouped_specialization" and k > 4096:
                    try:
                        library.launch(a, b, sa, sb, out, config, torch)
                    except RuntimeError:
                        rejected += 1
                        continue
                    raise AssertionError("Ungrouped experimental kernel accepted unsupported K")
                library.launch(a, b, sa, sb, out, config, torch)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                checked += 1
    print(f"{checked} launches passed full-output checks; {rejected} unsupported launches rejected")


if __name__ == "__main__":
    main()
