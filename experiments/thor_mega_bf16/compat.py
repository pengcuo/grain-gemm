"""Narrow provenance and numerical helpers for the Thor Mega experiment.

GrainGEMM benchmark code; upstream kernels stay external. The original source
pins and exact host-only replacements are checked before and after a run.
See LICENSES and README for the external DeepInfer/DeepGEMM dependencies.
No Torch import or device work occurs on import.
"""
import difflib
import hashlib
from pathlib import Path
import subprocess

DEEPGEMM_COMMIT = "1fc9c66975ee322d628b3288534ca0e9ce12a64d"
DEEPINFER_COMMIT = "8b16149741a4935e1dec8cea27853f273b374c5b"
COMPAT_HOST_GUARDS = {
    "csrc/apis/layout.hpp": (
        ("if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {",
         "if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 11)) {"),
        ("if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)",
         "if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 11))")),
    "csrc/apis/mega_moe.hpp": (("if (arch_major == 10) {", "if (arch_major == 10 or arch_major == 11) {"),
        ("if (arch_major == 10) {", "if (arch_major == 10 or (arch_major == 11 and jit->device.get_arch_minor() == 0)) {")),
}
COMPAT_SOURCE_ROOTS = {"csrc", "deep_gemm", "scripts", "docs", "tests", "setup.py"}


class SingleCardGroup:
    """Only the three methods used by upstream SymmBuffer's single-rank path."""
    def size(self):
        return 1

    def rank(self):
        return 0

    def barrier(self):
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def original_source_state(directory):
    names = subprocess.check_output(["git", "-C", str(directory), "ls-files", "-z"]).decode().split("\0")
    suffixes = {".py", ".hpp", ".h", ".cuh", ".cu", ".cpp", ".cc", ".json", ".cmake"}
    files = {name: sha(directory/name) for name in names if name and (directory/name).is_file()
             and (Path(name).suffix in suffixes or Path(name).name in {"CMakeLists.txt", ".gitmodules"})}
    submodules = {}
    for name in ("cutlass", "deep_jit"):
        relative = "third-party/"+name
        entry = git(directory,"ls-tree","HEAD","--",relative)
        actual = directory/relative
        submodules[relative] = dict(pinned_gitlink=entry.split()[2] if entry else None,
            actual_head=git(actual,"rev-parse","HEAD") if (actual/".git").exists() else None)
    return dict(path=str(directory), head=git(directory, "rev-parse", "HEAD"),
                tracked_diff=git(directory, "diff", "HEAD", "--"), source_sha256=files,
                submodule_pins=submodules)


def source_state(directory, original):
    pristine = original_source_state(original)
    if pristine["head"] != DEEPGEMM_COMMIT or pristine["tracked_diff"]:
        raise RuntimeError("Original provenance checkout must remain clean at the pinned commit")
    for relative,pins in pristine['submodule_pins'].items():
        if pins['pinned_gitlink'] != pins['actual_head'] or git(Path(original)/relative,'diff','HEAD','--'):
            raise RuntimeError('Pinned dependency changed: '+relative)
    names = subprocess.check_output(["git","-C",str(original),"ls-files","-z"]).decode().split("\0")
    files, changed = {}, {}
    for name in names:
        if not name or Path(name).parts[0] not in COMPAT_SOURCE_ROOTS or not (original/name).is_file():
            continue
        before, after = original/name, directory/name
        if not after.is_file():
            raise RuntimeError(f"Compatibility source is missing original tracked file: {name}")
        before_bytes, after_bytes = before.read_bytes(), after.read_bytes()
        files[name] = hashlib.sha256(after_bytes).hexdigest()
        if before_bytes == after_bytes:
            continue
        if name not in COMPAT_HOST_GUARDS:
            raise RuntimeError(f"Unexpected compatibility source modification: {name}")
        expected = before_bytes.decode()
        for old,new in COMPAT_HOST_GUARDS[name]:
            if old not in expected:
                raise RuntimeError(f"Expected original guard was not found: {name}: {old}")
            expected = expected.replace(old,new,1)
        if after_bytes != expected.encode():
            raise RuntimeError(f"Compatibility edit differs from the exact guard-only allowlist: {name}")
        changed[name] = dict(original_sha256=hashlib.sha256(before_bytes).hexdigest(),compat_sha256=files[name],
            unified_diff="".join(difflib.unified_diff(before_bytes.decode().splitlines(True),after_bytes.decode().splitlines(True),
                fromfile="original/"+name,tofile="sm110_compat/"+name)))
    if set(changed) != set(COMPAT_HOST_GUARDS):
        raise RuntimeError(f"Expected exactly the two compatibility files, found: {sorted(changed)}")
    tracked = set(files)
    unexpected = [str(path.relative_to(directory)) for path in directory.rglob("*")
                  if path.is_file() and path.suffix in {".py",".hpp",".h",".cuh",".cu",".cpp",".cc"}
                  and not str(path.relative_to(directory)).startswith("third-party/")
                  and str(path.relative_to(directory)) not in tracked]
    if unexpected:
        raise RuntimeError(f"Unexpected source files in isolated compatibility tree: {unexpected}")
    return dict(path=str(directory),original=pristine,compat_source_sha256=files,modified_files=changed,
                modified_file_count=len(changed),guard_edit_count=sum(len(v) for v in COMPAT_HOST_GUARDS.values()),
                copied_source_roots=sorted(COMPAT_SOURCE_ROOTS),
                experiment="SM110 host-guard compatibility, not unmodified upstream",
                kernel_arithmetic_layout_schedule_unchanged=True,actual_device_and_jit_architecture_preserved=True)


def package_payload_state(package, source):
    """Check all copied kernel/host/JIT headers used by the isolated extension."""
    mappings = ((source/"deep_gemm",package),(source/"csrc",package/"csrc"),
        (source/"third-party/cutlass/include/cute",package/"include/cute"),
        (source/"third-party/cutlass/include/cutlass",package/"include/cutlass"),
        (source/"third-party/deep_jit/include/deep_jit",package/"include/deep_jit"))
    checked = {}
    for root,destination in mappings:
        for original in sorted(root.rglob("*")):
            if not original.is_file() or original.suffix not in {".py",".hpp",".h",".cuh",".cu",".cpp",".cc"}:
                continue
            target = destination/original.relative_to(root)
            if not target.is_file() or sha(target) != sha(original):
                raise RuntimeError(f"Built compatibility payload differs from the verified source: {target}")
            checked[str(target.relative_to(package))] = sha(target)
    return dict(path=str(package),source_payload_sha256=checked,verified_files=len(checked))


def package_python_state(package, source):
    """Attest copied Python against source, permitting upstream-generated envs.py."""
    expected = {str(path.relative_to(source)):sha(path) for path in source.rglob("*.py")}
    actual = {str(path.relative_to(package)):sha(path) for path in package.rglob("*.py")}
    for name, expected_hash in expected.items():
        if actual.get(name) != expected_hash:
            raise RuntimeError(f"Loaded-package Python differs from original source: {package/name}")
    extras = set(actual)-set(expected)
    if extras-set(("envs.py",)):
        raise RuntimeError(f"Unexpected Python files in built package: {sorted(extras)}")
    return dict(path=str(package), source_path=str(source), python_sha256=actual,
                byte_identical_original_python_files=len(expected),
                generated_python_files={name:actual[name] for name in sorted(extras)},
                generated_note="envs.py may be generated by the original setup.py CustomBuildPy; it has no original source-file counterpart")


def cpu_reference(x_fp8, x_sf, ids, scores, raw_weights, torch, cast_fp8, unpack_sf,
                  activation_clamp):
    """Independent small CPU reference, retaining upstream round/score positions.

    Unlike upstream's optional DeepEP+TileLang legacy oracle, CPU FP32 matmul
    and exp may differ at rounding boundaries; this is a numerical check only.
    """
    x = x_fp8[:2].float().cpu()
    sf = unpack_sf(x_sf[:2].contiguous().cpu())[:, :x.shape[1]//32]
    x = (x.reshape(2, -1, 32)*sf[..., None]).reshape_as(x)
    ids, scores = ids[:2].cpu(), scores[:2].cpu()
    output, cache = torch.zeros_like(x), {}
    for token in range(2):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if expert not in cache:
                matrices = []
                for q, sf_float in raw_weights:
                    q_cpu, sf_cpu = q[expert].float().cpu(), sf_float[expert].cpu()
                    matrices.append((q_cpu.reshape(q_cpu.shape[0], -1, 32)*sf_cpu[..., None]).reshape_as(q_cpu))
                cache[expert] = matrices
            w1, w2 = cache[expert]
            gu = (x[token:token+1] @ w1.T).bfloat16().float()
            g, u = gu.chunk(2, -1)
            if activation_clamp is not None:
                g, u = g.clamp_max(activation_clamp), u.clamp(-activation_clamp, activation_clamp)
            hidden = ((g/(1+torch.exp(-g)))*u)*scores[token, slot]
            qhidden, sfhidden = cast_fp8(hidden, use_ue8m0=True, gran_k=32)
            restored = (qhidden.float().reshape(1, -1, 32)*sfhidden[..., None]).reshape_as(hidden)
            output[token] += (restored @ w2.T).bfloat16().float()[0]
    return output.bfloat16()


def common_fp32_reference(x,ids,scores,weights,torch):
    """Unquantized mathematical FFN from the original BF16 tensors, FP32 math."""
    x,ids,scores = x.float().cpu(),ids.cpu(),scores.cpu()
    cache = {}
    output = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token,slot])
            if expert not in cache:
                cache[expert] = tuple(w[expert].float().cpu() for w in weights)
            w1,w2 = cache[expert]
            gate,up = (x[token:token+1] @ w1.T).chunk(2,-1)
            hidden = torch.nn.functional.silu(gate)*up
            output[token] += (hidden @ w2.T)[0]*scores[token,slot]
    return output



def verify_deepinfer(directory):
    """Require the measured parent/submodule revisions, without loading CUDA."""
    directory=Path(directory).resolve()
    if git(directory,'rev-parse','HEAD')!=DEEPINFER_COMMIT or git(directory,'diff','HEAD','--'):
        raise RuntimeError('DeepInfer must be clean at '+DEEPINFER_COMMIT)
    original=directory/'vendor/DeepGEMM'
    entry=git(directory,'ls-tree','HEAD','--','vendor/DeepGEMM')
    if not entry or entry.split()[2]!=DEEPGEMM_COMMIT or git(original,'rev-parse','HEAD')!=DEEPGEMM_COMMIT:
        raise RuntimeError('DeepInfer vendor/DeepGEMM submodule must match the pinned gitlink')
    return dict(path=str(directory),head=DEEPINFER_COMMIT,deepgemm_gitlink=DEEPGEMM_COMMIT)
