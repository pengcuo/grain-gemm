"""Device checks for GB10-specific measurements, independent of runtime dispatch."""

import re


def require_gb10(torch, device=None):
    """Reject unsupported devices before loading native kernels or measuring.

    Importing this module does not import PyTorch or initialize CUDA, so report
    rendering, source audits, CLI help and cross-compilation remain CPU-only.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires NVIDIA GB10 (SM121) and a CUDA-enabled "
            "PyTorch installation; no CUDA device is available."
        )
    name = torch.cuda.get_device_name(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (12, 1) or re.search(r"\bGB10\b", name, re.IGNORECASE) is None:
        architecture = "sm_" + "".join(map(str, capability))
        raise RuntimeError(
            f"This experiment requires NVIDIA GB10 (SM121 / compute capability 12.1); "
            f"detected {name} ({architecture}). Use a separately validated and tuned "
            "experiment for Thor, H100, A100 or another GPU."
        )
    return {"name": name, "compute_capability": list(capability)}
