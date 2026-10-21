"""CPU-only checks for the optional fused library's provenance and launch ABI."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from grain_gemm.kernels import moe_fused


@pytest.fixture(autouse=True)
def clear_library_cache():
    moe_fused._load_library.cache_clear()
    yield
    moe_fused._load_library.cache_clear()


def _optional_build(tmp_path, monkeypatch, **overrides):
    library = tmp_path / "libgrain_moe_fused.so"
    library.write_bytes(b"test library")
    metadata = dict(architecture="sm_110", compiler_architecture="sm_110a",
                    abi_version=1, consumer_warps=[8, 16],
                    source_sha256=moe_fused.source_hashes(),
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    metadata.update(overrides)
    library.with_name("moe_fused_build.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(moe_fused, "_LIBRARY", library)
    return library


def test_absent_optional_build_is_unavailable(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(moe_fused, "_LIBRARY", tmp_path / "absent.so")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (11, 0))
    assert not moe_fused.is_available("cuda:3")


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (9, 0), (11, 1)])
def test_other_architectures_do_not_load_the_library(monkeypatch, capability):
    torch = pytest.importorskip("torch")
    device = object()

    def selected_capability(selected):
        assert selected is device
        return capability

    def unexpected_load():
        pytest.fail("An SM110-only library must not load on another architecture")

    monkeypatch.setattr(torch.cuda, "get_device_capability", selected_capability)
    monkeypatch.setattr(moe_fused, "_load_library", unexpected_load)
    assert not moe_fused.is_available(device)


@pytest.mark.parametrize("overrides", [
    {"architecture": "sm_120"},
    {"compiler_architecture": "sm_110"},
    {"abi_version": 0},
    {"consumer_warps": [16]},
    {"source_sha256": {}},
])
def test_incompatible_manifest_rejects_before_loading(tmp_path, monkeypatch, overrides):
    _optional_build(tmp_path, monkeypatch, **overrides)
    monkeypatch.setattr(moe_fused.ctypes, "CDLL", lambda path: pytest.fail("stale library loaded"))
    with pytest.raises(RuntimeError, match="stale or incompatible"):
        moe_fused._load_library()


def test_binary_must_match_manifest(tmp_path, monkeypatch):
    library = _optional_build(tmp_path, monkeypatch)
    library.write_bytes(b"replacement library")
    monkeypatch.setattr(moe_fused.ctypes, "CDLL", lambda path: pytest.fail("unmatched library loaded"))
    with pytest.raises(RuntimeError, match="does not match"):
        moe_fused._load_library()


@pytest.mark.parametrize("contents", ["not json", "null", "[]"])
def test_invalid_manifest_is_rejected(tmp_path, monkeypatch, contents):
    library = _optional_build(tmp_path, monkeypatch)
    library.with_name("moe_fused_build.json").write_text(contents)
    with pytest.raises(RuntimeError, match="metadata|incompatible"):
        moe_fused._load_library()


def test_valid_library_sets_independent_abi(tmp_path, monkeypatch):
    library_path = _optional_build(tmp_path, monkeypatch)
    library = SimpleNamespace(grain_moe_fused_launch=lambda *args: 0,
                              grain_moe_fused_smem_bytes=lambda warps: 167552,
                              grain_moe_fused_error_string=lambda code: b"error")
    paths = []

    def load(path):
        paths.append(path)
        return library

    monkeypatch.setattr(moe_fused.ctypes, "CDLL", load)
    assert moe_fused._load_library() is library
    assert moe_fused._load_library() is library
    assert paths == [str(library_path)]
    assert len(library.grain_moe_fused_launch.argtypes) == 14
    assert library.grain_moe_fused_launch.argtypes[:7] == [moe_fused.ctypes.c_void_p] * 7
    assert library.grain_moe_fused_launch.argtypes[7:13] == [moe_fused.ctypes.c_int] * 6


def test_prevalidated_launch_uses_each_calls_pointers_and_stream(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    streams = iter([71, 83])
    device = object()

    class Tensor:
        def __init__(self, ptr, shape=()):
            self.ptr, self.shape, self.device = ptr, shape, device

        def data_ptr(self):
            return self.ptr

    def current_stream(selected):
        assert selected is device
        return SimpleNamespace(cuda_stream=next(streams))

    def launch(*args):
        calls.append(args)
        return 0

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(moe_fused, "_load_library",
                        lambda: SimpleNamespace(grain_moe_fused_launch=launch))
    for first_pointer in (100, 200):
        a = Tensor(first_pointer, (32, 576, 1536))
        weight = Tensor(first_pointer + 1, (32, 2048, 1536))
        others = [Tensor(first_pointer + i) for i in range(2, 7)]
        out, scale_out = moe_fused._launch_prevalidated(
            a, weight, *others, consumer_warps=16, persistent_blocks=20)
        assert out is others[3] and scale_out is others[4]
    for first_pointer, stream, args in zip((100, 200), (71, 83), calls):
        assert args == (*range(first_pointer, first_pointer + 7),
                        576, 1024, 1536, 32, 16, 20, stream)


def test_native_launch_failure_is_reported(monkeypatch):
    torch = pytest.importorskip("torch")
    tensor = SimpleNamespace(shape=(32, 64, 1536), device=object(), data_ptr=lambda: 16)
    weight = SimpleNamespace(shape=(32, 2048, 1536), data_ptr=lambda: 32)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(moe_fused, "_load_library", lambda: SimpleNamespace(
        grain_moe_fused_launch=lambda *args: -10001,
        grain_moe_fused_error_string=lambda code: b"TMA descriptor rejected"))
    with pytest.raises(RuntimeError, match="TMA descriptor rejected"):
        moe_fused._launch_prevalidated(tensor, weight, tensor, tensor, tensor, tensor, tensor)
