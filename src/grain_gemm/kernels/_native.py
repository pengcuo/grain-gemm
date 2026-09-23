# SPDX-License-Identifier: Apache-2.0
"""Device targets supported by the optional native kernels."""

SUPPORTED_ARCHITECTURES = ("sm_120", "sm_121")


def architecture_for_device(device=None):
    """Resolve the actual input device, independently of the current device."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    architecture = f"sm_{major}{minor}"
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"Native kernels require SM120 or SM121; detected {architecture}"
        )
    return architecture
