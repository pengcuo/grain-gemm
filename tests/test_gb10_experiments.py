"""CPU-only checks for the GB10 experimental-device and build boundaries."""

import importlib.util
import hashlib
import json
import ctypes
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD_SCRIPTS = ("tools/build_cuda.py", "tools/build_cutlass.py")
BUILD_SCRIPTS = (
    "tools/build_cuda.py",
    "tools/build_cutlass.py",
    "experiments/cutlass/build.py",
    "experiments/cutlass/optimization/build.py",
)


@pytest.fixture
def require_gb10():
    # Import only the standard-library helper; these tests need neither PyTorch
    # nor a CUDA context, even on a machine that happens to have a GPU.
    path = ROOT / "benchmarks/_device.py"
    spec = importlib.util.spec_from_file_location("grain_benchmark_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_gb10


def fake_torch(name="NVIDIA GB10", capability=(12, 1), available=True):
    calls = []

    def is_available():
        calls.append(("available",))
        return available

    def get_name(device=None):
        if not available:
            pytest.fail("A missing CUDA runtime must be rejected before querying devices")
        calls.append(("name", device))
        return name

    def get_capability(device=None):
        if not available:
            pytest.fail("A missing CUDA runtime must be rejected before querying devices")
        calls.append(("capability", device))
        return capability

    cuda = SimpleNamespace(is_available=is_available, get_device_name=get_name,
                           get_device_capability=get_capability)
    return SimpleNamespace(cuda=cuda), calls


@pytest.mark.parametrize("name", ["NVIDIA GB10", "nvidia gb10", "NVIDIA GB10 Rev. 1"])
def test_gb10_device_is_accepted(require_gb10, name):
    torch, calls = fake_torch(name)
    detected = require_gb10(torch)
    assert detected["name"] == name
    assert tuple(detected["compute_capability"]) == (12, 1)
    assert ("name", None) in calls and ("capability", None) in calls


@pytest.mark.parametrize("name,capability", [
    ("NVIDIA Thor T5000", (11, 0)),
    ("NVIDIA H100", (9, 0)),
    ("NVIDIA GeForce RTX 5090", (12, 0)),
    ("Unmeasured NVIDIA device", (12, 1)),
    ("NVIDIA GB100", (12, 1)),
    ("NVIDIA GB10x", (12, 1)),
    ("NVIDIA GB10", (12, 0)),
])
def test_non_gb10_or_wrong_capability_is_rejected(require_gb10, name, capability):
    torch, _ = fake_torch(name, capability)
    with pytest.raises(RuntimeError) as failure:
        require_gb10(torch)
    message = str(failure.value)
    assert "GB10" in message and "SM121" in message
    assert name in message


def test_no_cuda_is_rejected_without_device_queries(require_gb10):
    torch, calls = fake_torch(available=False)
    with pytest.raises(RuntimeError, match="GB10"):
        require_gb10(torch)
    assert calls == [("available",)]


def test_explicit_device_is_used_for_both_queries(require_gb10):
    torch, calls = fake_torch()
    device = "cuda:2"
    require_gb10(torch, device=device)
    assert ("name", device) in calls and ("capability", device) in calls
    assert ("name", None) not in calls and ("capability", None) not in calls


def test_device_rejection_survives_python_optimization():
    # An assert would vanish under -O and silently admit another architecture.
    source = textwrap.dedent("""\
        import runpy
        import sys
        from types import SimpleNamespace
        require_gb10 = runpy.run_path(sys.argv[1])["require_gb10"]
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda device=None: "NVIDIA H100",
            get_device_capability=lambda device=None: (9, 0),
        )
        try:
            require_gb10(SimpleNamespace(cuda=cuda))
        except RuntimeError as error:
            print(error)
        else:
            raise SystemExit("GB10 restriction disappeared under python -O")
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-O", "-c", source,
         str(ROOT / "benchmarks/_device.py")],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GB10" in result.stdout and "SM121" in result.stdout
    assert "H100" in result.stdout


@pytest.fixture(params=BUILD_SCRIPTS)
def isolated_build(request, tmp_path):
    # Copy the real entrypoint so a regression cannot create/replace native
    # artifacts in the working checkout. Valid dummy headers ensure a missing
    # dependency cannot accidentally stand in for the architecture rejection.
    script = tmp_path / "repo" / request.param
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / request.param, script)
    headers = tmp_path / "cutlass"
    for relative in ("include/cute/tensor.hpp",
                     "include/cutlass/gemm/device/gemm_universal_adapter.h"):
        header = headers / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.touch()
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "Path(__file__).with_name('nvcc_was_called').touch()\n"
        "raise SystemExit(91)\n"
    )
    nvcc.chmod(0o755)
    return script, headers, nvcc, tmp_path


def tree_paths(root):
    return {path.relative_to(root) for path in root.rglob("*")}


@pytest.mark.parametrize("arch", ["sm_110", "sm_90", "sm_121a"])
def test_build_rejects_other_architectures_before_side_effects(isolated_build, arch):
    script, headers, nvcc, directory = isolated_build
    before = tree_paths(directory)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-O", str(script),
         "--cutlass-dir", str(headers), "--nvcc", str(nvcc), "--arch", arch],
        cwd=directory, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "sm_121" in result.stderr
    assert not (directory / "nvcc_was_called").exists()
    assert tree_paths(directory) == before, "Rejected architecture created build artifacts"


@pytest.mark.parametrize("isolated_build", BUILD_SCRIPTS[2:], indirect=True)
def test_gb10_experiment_builds_still_reject_sm120(isolated_build):
    script, headers, nvcc, directory = isolated_build
    before = tree_paths(directory)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-O", str(script),
         "--cutlass-dir", str(headers), "--nvcc", str(nvcc), "--arch", "sm_120"],
        cwd=directory, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "GB10" in result.stderr and "sm_121" in result.stderr
    assert not (directory / "nvcc_was_called").exists()
    assert tree_paths(directory) == before


@pytest.mark.parametrize("isolated_build", NATIVE_BUILD_SCRIPTS, indirect=True)
@pytest.mark.parametrize("arch", ["sm_120", "sm_121"])
def test_native_build_records_requested_target_without_gpu(isolated_build, arch):
    script, headers, nvcc, directory = isolated_build
    source = directory / "repo/src/grain_gemm/kernels/csrc"
    source.mkdir(parents=True)
    contents = {
        "sm12x/int8_g256_cute.cu": "// isolated CuTe source\n",
        "sm12x/cutlass/int8_g256.cu": "// isolated CUTLASS source\n",
        "sm12x/cutlass/collective.hpp": "// isolated collective source\n",
        "sm12x/cutlass/epilogue.hpp": "// isolated epilogue source\n",
        "sm12x/cutlass/kernel.hpp": "// isolated kernel source\n",
    }
    for name, content in contents.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(content)
    nvcc.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import json, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('Fake CUDA compiler for CPU-only build contract test')\n"
        "    raise SystemExit(0)\n"
        "Path(__file__).with_name('compiler_args.json').write_text(json.dumps(sys.argv[1:]))\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'test-native-library')\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(script), "--cutlass-dir", str(headers),
         "--nvcc", str(nvcc), "--arch", arch],
        cwd=directory, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    flags = json.loads((directory / "compiler_args.json").read_text())
    assert f"-arch={arch}" in flags
    native = directory / "repo/src/grain_gemm/kernels/_native"
    backend = "cuda" if script.name == "build_cuda.py" else "cutlass"
    manifests = list(native.rglob("build.json" if backend == "cuda" else "cutlass_build.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["architecture"] == arch
    if backend == "cuda":
        expected = hashlib.sha256(contents["sm12x/int8_g256_cute.cu"].encode()).hexdigest()
    else:
        expected = {Path(name).name: hashlib.sha256(content.encode()).hexdigest()
                    for name, content in contents.items()
                    if name.startswith("sm12x/cutlass/")}
    assert manifest["source_sha256"] == expected
    library = manifests[0].with_name(f"libgrain_{backend}.so")
    assert library.read_bytes() == b"test-native-library"


def test_build_help_needs_no_cuda_or_optional_dependencies(isolated_build):
    script, _, _, directory = isolated_build
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    before = tree_paths(directory)
    result = subprocess.run(
        # -S excludes site-packages, including PyTorch and Triton.
        [sys.executable, "-S", "-B", str(script), "--help"],
        cwd=directory, env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--arch" in result.stdout and "sm_121" in result.stdout
    assert tree_paths(directory) == before


@pytest.mark.parametrize("entrypoint", ["paired_benchmark.py", "run.py"])
def test_measurement_cli_rejects_thor_before_native_loading_or_writing(
        monkeypatch, tmp_path, entrypoint):
    torch, _ = fake_torch("NVIDIA Thor T5000", (11, 0))
    monkeypatch.setitem(sys.modules, "torch", torch)
    experiment = ROOT / "experiments/cutlass"
    monkeypatch.setattr(sys, "path", [str(experiment), *sys.path])
    script = experiment / entrypoint
    output = tmp_path / "measurements"
    arguments = [str(script), "--output", str(output)]
    if entrypoint == "paired_benchmark.py":
        arguments += ["--case", "1024", "1024", "1536", "3", "11"]
    monkeypatch.setattr(sys, "argv", arguments)

    def unexpected_native_action(*args, **kwargs):
        pytest.fail("An unsupported GPU reached native loading or a child measurement")

    monkeypatch.setattr(ctypes, "CDLL", unexpected_native_action)
    monkeypatch.setattr(subprocess, "run", unexpected_native_action)
    with pytest.raises(RuntimeError, match="GB10.*SM121.*Thor"):
        runpy.run_path(str(script), run_name="__main__")
    assert not output.exists()


def test_paired_dry_run_needs_no_torch_and_creates_no_results(tmp_path):
    output = tmp_path / "measurements"
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(ROOT / "experiments/cutlass/run.py"),
         "--suite", "m4096", "--output", str(output), "--dry-run"],
        cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout.splitlines()
    assert len(commands) == 4
    assert all("--case 4096 " in command for command in commands)
    assert not output.exists()
