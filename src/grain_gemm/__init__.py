"""GrainGEMM: INT8 GEMM with K-group scaling."""

__version__ = "0.0.0"
__all__ = ["int8_gemm", "get_kernel_config", "__version__"]


def __getattr__(name):
    # Keep package/version inspection usable without the optional GPU runtime.
    if name in ("int8_gemm", "get_kernel_config"):
        from .gemm import int8_gemm, get_kernel_config
        globals().update(int8_gemm=int8_gemm, get_kernel_config=get_kernel_config)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
