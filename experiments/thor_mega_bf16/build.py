#!/usr/bin/env python3
"""Prepare the pinned four-guard Thor Mega source; optionally build its host extension.

No GPU kernel is compiled or executed by this script. Device JIT compilation
happens only when a caller invokes the prepared DeepGEMM package.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PINS = dict(deepinfer='8b16149741a4935e1dec8cea27853f273b374c5b',
    deep_gemm='1fc9c66975ee322d628b3288534ca0e9ce12a64d',
    cutlass='f3fde58372d33e9a5650ba7b80fc48b3b49d40c8',
    deep_jit='2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def copy_tracked(source, destination, *, headers_only=False):
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    copied = {}
    for name in names:
        if not name:
            continue
        relative = Path(name)
        if headers_only and relative.parts[0] != 'include' and not relative.name.startswith('LICENSE'):
            continue
        original = source/relative
        if not original.is_file():  # Submodule gitlinks are copied separately.
            continue
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target, follow_symlinks=True)
        copied[name] = sha(target)
        if copied[name] != sha(original):
            raise RuntimeError('Source changed while copying: '+str(original))
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepinfer-dir', type=Path, required=True, help='Original pinned DeepInfer checkout')
    parser.add_argument('--source-dir', type=Path, required=True, help='New isolated patched-source directory')
    parser.add_argument('--build-dir', type=Path, required=True, help='New package/log/manifest directory')
    parser.add_argument('--build', action='store_true', help='Build host C++ extension and package; never execute GPU work')
    parser.add_argument('--build-deps-dir', type=Path, help='Optional directory added to build PYTHONPATH')
    parser.add_argument('--extra-include-dir', type=Path, help='Optional supplemental host headers added to build CPATH')
    parser.add_argument('--dry-run', action='store_true', help='Verify pins and print the plan without writing/building')
    args = parser.parse_args()
    # Keep --help independent of PyTorch/CUDA and of the helper module.
    sys.path.insert(0, str(HERE))
    import compat
    deepinfer, source, build = (p.expanduser().resolve() for p in
        (args.deepinfer_dir, args.source_dir, args.build_dir))
    original = deepinfer/'vendor/DeepGEMM'
    for target in (source, build):
        if target.exists():
            parser.error('Require a new output directory: '+str(target))
        if target.is_relative_to(deepinfer):
            parser.error('Output must be outside the original checkout: '+str(target))
    if source == build or source.is_relative_to(build) or build.is_relative_to(source):
        parser.error('Source and build directories must be separate')
    compat.verify_deepinfer(deepinfer)
    original_state = compat.original_source_state(original)
    repositories = dict(deepinfer=deepinfer, deep_gemm=original,
        cutlass=original/'third-party/cutlass', deep_jit=original/'third-party/deep_jit')
    for name, repository in repositories.items():
        if git(repository, 'rev-parse', 'HEAD') != PINS[name]:
            raise RuntimeError(f'Require pinned {name} commit {PINS[name]}')
        if git(repository, 'diff', 'HEAD', '--'):
            raise RuntimeError('Require clean tracked source: '+str(repository))
    command = [sys.executable, 'setup.py', 'build_ext', '--build-lib', str(build),
        '--build-temp', str(build/'temp'), 'build_py', '--build-lib', str(build)]
    plan = dict(pins=PINS, original=str(original), source=str(source), package=str(build/'deep_gemm'),
        command=command, working_directory=str(source), build_requested=args.build,
        environment_overrides=dict(DG_USE_LOCAL_VERSION='0', DG_FORCE_BUILD='1'),
        guard_changes=4, device_algorithm_changed=False, gpu_execution=False)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    source.mkdir(parents=True)
    build.mkdir(parents=True)
    original_files = copy_tracked(original, source)
    dependency_files = {name: copy_tracked(repositories[name], source/'third-party'/name,
        headers_only=True) for name in ('cutlass', 'deep_jit')}
    for name, replacements in compat.COMPAT_HOST_GUARDS.items():
        path = source/name
        text = path.read_text()
        for before, after in replacements:
            if before not in text:
                raise RuntimeError('Expected original host guard missing: '+name)
            text = text.replace(before, after, 1)
        path.write_text(text)
    attestation = compat.source_state(source, original)
    if attestation['guard_edit_count'] != 4:
        raise RuntimeError('Expected exactly four host architecture-guard edits')
    manifest = dict(plan, original_state=original_state, source_attestation=attestation,
        original_copied_files=original_files, dependency_copied_files=dependency_files,
        patch_sha256=sha(HERE/'compatibility.patch'),
        builder_sha256=sha(__file__), host_build_completed=False)
    manifest_path = build/'compatibility-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.build:
        environment = dict(os.environ, DG_USE_LOCAL_VERSION='0', DG_FORCE_BUILD='1')
        for argument, variable in ((args.build_deps_dir, 'PYTHONPATH'), (args.extra_include_dir, 'CPATH')):
            if argument is not None:
                path = argument.expanduser().resolve()
                if not path.is_dir():
                    raise ValueError('Missing optional dependency directory: '+str(path))
                environment[variable] = str(path) + (os.pathsep+environment[variable] if environment.get(variable) else '')
        with (build/'build.log').open('w') as log:
            subprocess.run(command, cwd=source, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True)
        package = build/'deep_gemm'
        # Upstream setup.py copies CUTLASS but omits this required JIT header tree.
        shutil.copytree(source/'third-party/deep_jit/include/deep_jit', package/'include/deep_jit')
        for name, files in dependency_files.items():
            if any(sha(source/'third-party'/name/relative) != expected for relative, expected in files.items()):
                raise RuntimeError('Dependency headers changed during build: '+name)
        manifest.update(host_build_completed=True,
            source_attestation=compat.source_state(source, original),
            package_payload=compat.package_payload_state(package, source),
            package_python=compat.package_python_state(package, source/'deep_gemm'),
            native_sha256={str(p.relative_to(package)):sha(p) for p in package.glob('*.so')},
            build_log_sha256=sha(build/'build.log'), build_environment={name:environment.get(name)
                for name in ('DG_USE_LOCAL_VERSION','DG_FORCE_BUILD','PYTHONPATH','CPATH')})
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(source=str(source), package=str(build/'deep_gemm'),
        manifest=str(manifest_path), host_build_completed=args.build, gpu_execution=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
