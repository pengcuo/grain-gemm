"""CPU-only checks for the optional original-token library's provenance and launch ABI."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from grain_gemm.kernels import moe_gather


@pytest.fixture(autouse=True)
def clear_library_cache():
    moe_gather._load_library.cache_clear()
    yield
    moe_gather._load_library.cache_clear()


def _optional_build(tmp_path, monkeypatch, **overrides):
    library = tmp_path / "libgrain_moe_gather.so"
    library.write_bytes(b"test library")
    metadata = dict(architecture="sm_110", compiler_architecture="sm_110a",
                    abi_version=1, consumer_warps=[8, 16],
                    source_sha256=moe_gather.source_hashes(),
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    metadata.update(overrides)
    library.with_name("moe_gather_build.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(moe_gather, "_LIBRARY", library)
    return library


def test_absent_optional_build_is_unavailable(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(moe_gather, "_LIBRARY", tmp_path / "absent.so")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (11, 0))
    assert not moe_gather.is_available("cuda:3")


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
    monkeypatch.setattr(moe_gather, "_load_library", unexpected_load)
    assert not moe_gather.is_available(device)


@pytest.mark.parametrize("overrides", [
    {"architecture": "sm_120"},
    {"compiler_architecture": "sm_110"},
    {"abi_version": 0},
    {"consumer_warps": [16]},
    {"source_sha256": {}},
])
def test_incompatible_manifest_rejects_before_loading(tmp_path, monkeypatch, overrides):
    _optional_build(tmp_path, monkeypatch, **overrides)
    monkeypatch.setattr(moe_gather.ctypes, "CDLL", lambda path: pytest.fail("stale library loaded"))
    with pytest.raises(RuntimeError, match="stale or incompatible"):
        moe_gather._load_library()


def test_binary_must_match_manifest(tmp_path, monkeypatch):
    library = _optional_build(tmp_path, monkeypatch)
    library.write_bytes(b"replacement library")
    monkeypatch.setattr(moe_gather.ctypes, "CDLL", lambda path: pytest.fail("unmatched library loaded"))
    with pytest.raises(RuntimeError, match="does not match"):
        moe_gather._load_library()


@pytest.mark.parametrize("contents", ["not json", "null", "[]"])
def test_invalid_manifest_is_rejected(tmp_path, monkeypatch, contents):
    library = _optional_build(tmp_path, monkeypatch)
    library.with_name("moe_gather_build.json").write_text(contents)
    with pytest.raises(RuntimeError, match="metadata|incompatible"):
        moe_gather._load_library()


def test_valid_library_sets_independent_abi(tmp_path, monkeypatch):
    library_path = _optional_build(tmp_path, monkeypatch)
    library = SimpleNamespace(grain_moe_gather_launch=lambda *args: 0,
                              grain_moe_gather_smem_bytes=lambda warps: 167552,
                              grain_moe_gather_error_string=lambda code: b"error")
    paths = []

    def load(path):
        paths.append(path)
        return library

    monkeypatch.setattr(moe_gather.ctypes, "CDLL", load)
    assert moe_gather._load_library() is library
    assert moe_gather._load_library() is library
    assert paths == [str(library_path)]
    assert len(library.grain_moe_gather_launch.argtypes) == 18
    assert library.grain_moe_gather_launch.argtypes[:9] == [moe_gather.ctypes.c_void_p] * 9
    assert library.grain_moe_gather_launch.argtypes[9:17] == [moe_gather.ctypes.c_int] * 8


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
    monkeypatch.setattr(moe_gather, "_load_library",
                        lambda: SimpleNamespace(grain_moe_gather_launch=launch))
    for first_pointer in (100, 200):
        x = Tensor(first_pointer, (4096, 1536))
        weight = Tensor(first_pointer + 1, (32, 2048, 1536))
        others = [Tensor(first_pointer + i) for i in range(2, 9)]
        others[5].shape = (32, 576, 1024)
        out, scale_out = moe_gather._launch_prevalidated(
            x, weight, *others, top_k=4, consumer_warps=16, persistent_blocks=20)
        assert out is others[5] and scale_out is others[6]
    for first_pointer, stream, args in zip((100, 200), (71, 83), calls):
        expected = [first_pointer+i for i in (0,2,1,3,4,5,6,7,8)]
        assert args == (*expected,576,1024,1536,32,4096,4,16,20,stream)


def test_native_launch_failure_is_reported(monkeypatch):
    torch = pytest.importorskip("torch")
    tensor = SimpleNamespace(shape=(32, 64, 1536), device=object(), data_ptr=lambda: 16)
    weight = SimpleNamespace(shape=(32, 2048, 1536), data_ptr=lambda: 32)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(moe_gather, "_load_library", lambda: SimpleNamespace(
        grain_moe_gather_launch=lambda *args: -10001,
        grain_moe_gather_error_string=lambda code: b"TMA descriptor rejected"))
    with pytest.raises(RuntimeError, match="TMA descriptor rejected"):
        moe_gather._launch_prevalidated(tensor, weight, tensor, tensor, tensor, tensor, tensor, tensor, tensor, top_k=4)
