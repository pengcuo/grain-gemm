# SPDX-License-Identifier: Apache-2.0
"""Device targets supported by the optional native kernels."""

SUPPORTED_ARCHITECTURES = ("sm_110", "sm_120", "sm_121")


def architecture_for_device(device=None):
    """Resolve the actual input device, independently of the current device."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    architecture = f"sm_{major}{minor}"
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"Native kernels require SM110, SM120 or SM121; detected {architecture}"
        )
    return architecture


def cuda_sources(architecture):
    """Translation unit and transitive project dependencies for source checking."""
    from pathlib import Path

    root = Path(__file__).parent / "csrc"
    legacy = root / "sm12x" / "int8_g256_cute.cu"
    if architecture == "sm_110":
        target = root / "sm110"
        return (target / "int8_g256_cute.cu",
                target / "int8_g256_pipeline.cuh",
                target / "int8_g256_pipeline_1sm.cuh", legacy)
    return (legacy,)


def cuda_source_hash(architecture):
    import hashlib

    sources = cuda_sources(architecture)
    # Preserve the GB10 build metadata contract. Thor also hashes its dependency.
    if len(sources) == 1:
        return hashlib.sha256(sources[0].read_bytes()).hexdigest()
    return {str(p.relative_to(sources[0].parents[1])):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def cutlass_sources(architecture):
    """CUTLASS translation unit and every transitive project source dependency."""
    from pathlib import Path

    root = Path(__file__).parent / "csrc"
    legacy = root / "sm12x" / "cutlass"
    dependencies = tuple(path for path in sorted(legacy.iterdir())
                         if path.suffix in (".cu", ".hpp"))
    if architecture == "sm_110":
        target = root / "sm110"
        shadows = tuple(sorted((target / "include").rglob("*")))
        return (target / "int8_g256_cutlass.cu",
                *(path for path in shadows if path.suffix in (".h", ".hpp")),
                *dependencies)
    return (legacy / "int8_g256.cu",
            *(path for path in dependencies if path.name != "int8_g256.cu"))


def cutlass_source_hash(architecture):
    """Keep GB10's basename manifest while hashing Thor's shadow headers too."""
    import hashlib
    from pathlib import Path

    root = Path(__file__).parent / "csrc"
    return {
        (str(path.relative_to(root)) if architecture == "sm_110" else path.name):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in cutlass_sources(architecture)
    }
