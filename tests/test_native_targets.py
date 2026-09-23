"""CPU-only coverage for opt-in SM120 native builds and target isolation."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from grain_gemm import get_kernel_config
from test_cutlass_gemm import empty_dispatch_inputs


@pytest.fixture(params=["cuda", "cutlass"])
def native(request, monkeypatch, tmp_path):
    module = importlib.import_module(f"grain_gemm.kernels.{request.param}")
    loader = module._load_library
    loader.cache_clear()
    library = tmp_path / f"libgrain_{request.param}.so"
    library.touch()
    monkeypatch.setattr(module, "_LIBRARY", library)
    yield module
    loader.cache_clear()


def write_metadata(native, architecture, *, stale=False):
    if native.__name__.endswith(".cuda"):
        manifest = "build.json"
        source = Path(native.__file__).parent / "csrc/sm12x/int8_g256_cute.cu"
        hashes = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        manifest = "cutlass_build.json"
        hashes = native._source_hashes()
    native._LIBRARY.with_name(manifest).write_text(json.dumps({
        "architecture": architecture,
        "source_sha256": "stale" if stale else hashes,
    }))


def fake_library():
    # CDLL symbols allow argtypes/restype attributes to be assigned by the loader.
    def symbol():
        return lambda *args: 0
    return SimpleNamespace(grain_cute_g256=symbol(), grain_cuda_error_string=symbol(),
                           grain_cutlass_g256=symbol(), grain_cutlass_error_string=symbol())


@pytest.mark.parametrize("architecture", ["sm_120", "sm_121"])
def test_matching_native_target_loads(native, monkeypatch, architecture):
    write_metadata(native, architecture)
    library = fake_library()
    calls = []
    monkeypatch.setattr(native.ctypes, "CDLL", lambda path: calls.append(path) or library)
    assert native._load_library(architecture) is library
    assert native._load_library(architecture) is library
    assert calls == [str(native._LIBRARY)]


@pytest.mark.parametrize("built,requested", [("sm_121", "sm_120"),
                                              ("sm_120", "sm_121")])
def test_mismatched_build_rejected_before_cdll(native, monkeypatch, built, requested):
    write_metadata(native, built)
    monkeypatch.setattr(native.ctypes, "CDLL",
                        lambda *args: pytest.fail("Mismatched target reached CDLL"))
    with pytest.raises(RuntimeError) as failure:
        native._load_library(requested)
    message = str(failure.value)
    assert built in message and requested in message
    assert f"--arch {requested}" in message


def test_target_cache_does_not_reuse_library_for_another_device(native, monkeypatch):
    write_metadata(native, "sm_121")
    library = fake_library()
    calls = []
    monkeypatch.setattr(native.ctypes, "CDLL", lambda path: calls.append(path) or library)
    assert native._load_library("sm_121") is library
    # Warming the cache for the current GB10 cannot admit a requested RTX device.
    with pytest.raises(RuntimeError, match="sm_120"):
        native._load_library("sm_120")
    assert native._load_library("sm_121") is library
    assert len(calls) == 1


@pytest.mark.parametrize("architecture", ["sm_120", "sm_121"])
def test_stale_sources_rejected_before_cdll(native, monkeypatch, architecture):
    write_metadata(native, architecture, stale=True)
    monkeypatch.setattr(native.ctypes, "CDLL",
                        lambda *args: pytest.fail("Stale source reached CDLL"))
    with pytest.raises(RuntimeError, match="stale"):
        native._load_library(architecture)


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_availability_resolves_requested_device(native, monkeypatch, capability):
    device = torch.device("cuda:3")
    queried = []
    targets = []
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda selected=None: queried.append(selected) or capability)
    monkeypatch.setattr(native, "_load_library", lambda target: targets.append(target))
    assert native.is_available(device=device)
    assert queried == [device]
    assert targets == [f"sm_{capability[0]}{capability[1]}"]


def test_unavailable_target_does_not_load_native_library(native, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    monkeypatch.setattr(native, "_load_library",
                        lambda *args: pytest.fail("Unsupported architecture reached loader"))
    assert native.is_available(device=torch.device("cuda:1")) is False


def test_direct_launch_checks_tensor_device_before_loading(native, monkeypatch):
    device = torch.device("cuda:2")
    queried = []
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda selected=None: queried.append(selected) or (12, 0))
    write_metadata(native, "sm_121")
    monkeypatch.setattr(native.ctypes, "CDLL",
                        lambda *args: pytest.fail("Mismatched direct launch reached CDLL"))
    with pytest.raises(RuntimeError, match="sm_120"):
        native.launch(SimpleNamespace(device=device), None, None, None, None)
    assert queried == [device]


@pytest.mark.parametrize("shape", [(256, 1024, 1536), (1024, 1024, 1024),
                                   (128, 640, 768)])
@pytest.mark.parametrize("backend", ["cuda", "cutlass"])
def test_sm120_explicit_ignores_gb10_measurements(monkeypatch, shape, backend):
    import grain_gemm.gemm as implementation
    module = importlib.import_module(f"grain_gemm.kernels.{backend}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    queried = []
    monkeypatch.setattr(module, "is_available", lambda device=None: queried.append(device) or True)
    # Deliberately incompatible measured IDs make accidental reuse observable.
    entry = {"selected": {"backend": "cuda", "config_id": 16},
             "cuda": {"backend": "cuda", "config_id": 16},
             "cutlass": {"backend": "cutlass", "config_id": 31},
             "triton": {"backend": "triton", "block_m": 128, "block_n": 128}}
    key = "x".join(map(str, shape))
    monkeypatch.setattr(implementation, "_dispatch_table", lambda: {
        "square_configs": {str(shape[0]): entry}, "rectangular_configs": {key: entry}})
    monkeypatch.setattr(implementation, "_cutlass_dispatch_table", lambda: {
        "configs": {key: {"config": {"backend": "cutlass", "config_id": 31}}}})
    a, b, sa, sb = empty_dispatch_inputs(*shape)
    config = get_kernel_config(a, b, scale_a=sa, scale_b=sb, backend=backend)
    assert config["backend"] == backend
    assert config["config_id"] == 0
    assert config["compute_capability"] == [12, 0]
    suffix = "cutlass_explicit" if backend == "cutlass" else "explicit"
    assert config["policy"] == f"sm120_g256_{suffix}"
    assert queried == [a.device]


@pytest.mark.parametrize("available", [False, True])
def test_sm120_auto_stays_portable_with_or_without_native_build(monkeypatch, available):
    from grain_gemm.kernels import cuda, cutlass
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(cuda, "is_available", lambda device=None: available)
    monkeypatch.setattr(cutlass, "is_available", lambda device=None: available)
    a, b, sa, sb = empty_dispatch_inputs(256, 1024, 1536)
    config = get_kernel_config(a, b, scale_a=sa, scale_b=sb)
    assert config["backend"] == "triton"
    assert config["policy"] == "portable"


@pytest.mark.parametrize("backend", ["cuda", "cutlass"])
def test_sm120_missing_build_has_target_specific_guidance(monkeypatch, backend):
    module = importlib.import_module(f"grain_gemm.kernels.{backend}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(module, "is_available", lambda device=None: False)
    a, b, sa, sb = empty_dispatch_inputs(256, 1024, 1536)
    with pytest.raises(RuntimeError, match="--arch sm_120"):
        get_kernel_config(a, b, scale_a=sa, scale_b=sb, backend=backend)
