#!/usr/bin/env python3
"""CPU-only actual patch transaction + local-only CMake acquisition upgrade tests.
No native compilation, network, GPU, weights, system cache, or external Git writes.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream-source', type=Path, required=True)
p.add_argument('--legacy-source', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
a = p.parse_args()
a.work_dir.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
paths = [line.removeprefix('source:') for line in (root/'patches/native-upstream-baseline.txt').read_text().splitlines() if line.startswith('source:')]
def run(command, cwd=None, good=True, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    if good:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, 'negative unexpectedly passed: ' + str(command)
    return result

def copy_sources(source, target):
    target.mkdir()
    for name in paths:
        src = source/name
        if src.exists():
            dst = target/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

def snapshot(directory):
    return {str(f.relative_to(directory)): (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns)
            for f in directory.rglob('*') if f.is_file()}

binary = a.work_dir/'patch-driver'
run(['nice', '-n', '10', str(Path.home()/'.cargo/bin/rustc'), '--edition=2021', '-C', 'debuginfo=0', '-C', 'strip=none',
     str(root/'tests/patch_series_driver.rs'), '-o', str(binary)])
patches = a.work_dir/'patches'
run([str(binary), str(root), str(patches), 'aabbcc'])
script = patches/'apply_patches.sh'
fresh = a.work_dir/'fresh-no-git'
copy_sources(a.upstream_source, fresh)
run(['nice', '-n', '10', 'sh', str(script)], cwd=fresh)
first = snapshot(fresh)
result = run(['nice', '-n', '10', 'sh', str(script)], cwd=fresh)
assert 'no-op reentry' in result.stderr
assert snapshot(fresh) == first, 'reentry must not touch source or receipt mtimes'
print('PASS: actual 27-patch fresh no.git application + exact no-op reentry', flush=True)

for case in ['corrupt-source', 'missing-source', 'corrupt-receipt', 'partial-no-receipt', 'introduced-no-receipt', 'changed-patch-input', 'changed-order']:
    target = a.work_dir/case
    source = fresh if case not in ['partial-no-receipt', 'introduced-no-receipt'] else a.upstream_source
    if source == fresh:
        shutil.copytree(source, target)
    else:
        copy_sources(source, target)
    selected_script = script
    if case == 'corrupt-source':
        with (target/'mlx/backend/metal/device.cpp').open('a') as f: f.write('\n// corruption\n')
    elif case == 'missing-source':
        (target/'mlx/backend/metal/completion_retirement.h').unlink()
    elif case == 'corrupt-receipt':
        (target/'.pmetal-patch-series-v1').write_text('success\n')
    elif case == 'partial-no-receipt':
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(target.resolve()))
        run(['patch', '-s', '-p1', '-t', '-N', '-i', str(patches/'metallib-search-path.patch')], cwd=target, env=env)
    elif case == 'introduced-no-receipt':
        f = target/'mlx/backend/metal/completion_retirement.h'
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('// partial introduced source\n')
    elif case == 'changed-patch-input':
        changed = a.work_dir/'changed-patches'
        shutil.copytree(patches, changed)
        with (changed/'completion-retirement.patch').open('a') as f: f.write('\n')
        selected_script = changed/'apply_patches.sh'
    elif case == 'changed-order':
        changed_root = a.work_dir/'reordered-sys'
        changed_root.mkdir()
        shutil.copytree(root/'patches', changed_root/'patches')
        text = (root/'build.rs').read_text()
        text = text.replace('"patches/metallib-search-path.patch",', '"patches/TEMP.patch",', 1)
        text = text.replace('"patches/command-buffer-recoverable.patch",', '"patches/metallib-search-path.patch",', 1)
        text = text.replace('"patches/TEMP.patch",', '"patches/command-buffer-recoverable.patch",', 1)
        (changed_root/'build.rs').write_text(text)
        changed = a.work_dir/'reordered-patches'
        run([str(binary), str(changed_root), str(changed), 'ddeeff'])
        selected_script = changed/'apply_patches.sh'
    before = snapshot(target)
    result = run(['nice', '-n', '10', 'sh', str(selected_script)], cwd=target, good=False)
    assert 'native patch integrity:' in result.stderr, result.stderr
    assert snapshot(target) == before, 'negative must refuse without mutating source/receipt'
    print('PASS: discriminating rejection: ' + case, flush=True)

# Regression: patch -p1 accepts arbitrary x/y prefixes. They MUST NOT evade
# source-path accounting merely because the supported parser uses a/b headers.
unsupported = '--- x/extra.cpp\n+++ y/extra.cpp\n@@ -1 +1 @@\n-before\n+after\n'
applicable = a.work_dir/'otherwise-applicable-section'
applicable.mkdir()
(applicable/'extra.cpp').write_text('before\n')
unsupported_patch = applicable/'unsupported.patch'
unsupported_patch.write_text(unsupported)
run(['patch', '-s', '-p1', '-t', '-N', '-i', str(unsupported_patch)], cwd=applicable)
assert (applicable/'extra.cpp').read_text() == 'after\n', 'negative must be an actually applicable file section'
bad_root = a.work_dir/'unsupported-section-sys'
bad_root.mkdir()
shutil.copytree(root/'patches', bad_root/'patches')
(bad_root/'build.rs').write_text((root/'build.rs').read_text())
with (bad_root/'patches/metallib-search-path.patch').open('a') as f:
    f.write('\n'+unsupported)
result = run([str(binary), str(bad_root), str(a.work_dir/'unsupported-section-output'), 'aabbcc'], good=False)
assert 'unsupported native patch path/header' in result.stderr, result.stderr
print('PASS: rejects otherwise-applicable unsupported-prefix section before emitting/applying transaction', flush=True)

# An old valid fully patched tree has no new receipt: it is not silently blessed.
# The actual CMake declaration routes an upgrade into a NEW owned source/binary
# path, preserving the old tree. Exercise real FetchContent acquisition locally.
legacy = a.work_dir/'known-legacy'
copy_sources(a.legacy_source, legacy)
legacy_before = snapshot(legacy)
run(['nice', '-n', '10', 'sh', str(script)], cwd=legacy, good=False)
assert snapshot(legacy) == legacy_before

fixture = a.work_dir/'cmake-fixture'
fixture.mkdir()
repo = fixture/'upstream'
repo.mkdir()
(repo/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(native_fixture NONE)\n')
(repo/'a.txt').write_text('before\n')
run(['git', 'init', '-q', str(repo)])
run(['git', '-C', str(repo), 'add', 'CMakeLists.txt', 'a.txt'])
run(['git', '-C', str(repo), '-c', 'user.name=CPU Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-q', '-m',
     'test fixture\n\nCo-Authored-By: Weave Router <router@weaveos.com>'])
run(['git', '-C', str(repo), 'tag', 'v0.32.0'])
fixture_sys = fixture/'sys'
(fixture_sys/'patches').mkdir(parents=True)
(fixture_sys/'build.rs').write_text('let patch_files = [\n    "patches/change.patch",\n    ];\n')
(fixture_sys/'patches/change.patch').write_text('diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-before\n+after\n')
(fixture_sys/'patches/native-upstream-baseline.txt').write_text('source:a.txt\n' + hashlib.sha256(b'before\n').hexdigest()+'\n')
(fixture_sys/'patches/native-patched-baseline.txt').write_text('source:a.txt\n' + hashlib.sha256(b'after\n').hexdigest()+'\n')
project = fixture/'project'
project.mkdir()
fixture_patches = project/'patches'
run([str(binary), str(fixture_sys), str(fixture_patches), 'aabbcc'])
# Stage the OLD acquisition shape, patched but without our new receipt.
legacy_script = fixture_patches/'legacy.sh'
legacy_script.write_text('set -eu\ngit apply "$(dirname "$0")/change.patch"\n')
def configure(declaration):
    (project/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(upgrade NONE)\ninclude(FetchContent)\nFetchContent_Declare(mlx\n    GIT_REPOSITORY "'+str(repo)+'"\n    '+declaration+'\nFetchContent_MakeAvailable(mlx)\n')
    run(['nice', '-n', '10', 'cmake', '-S', str(project), '-B', str(fixture/'build')])
configure('GIT_TAG v0.32.0\n    PATCH_COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/patches/legacy.sh)')
old_source = fixture/'build/_deps/mlx-src'
assert (old_source/'a.txt').read_text() == 'after\n'
assert not (old_source/'.pmetal-patch-series-v1').exists()
old_before = snapshot(old_source)
configure((fixture_patches/'cmake-declaration.txt').read_text())
new_source = fixture/'build/_deps/pmetal-aabbcc/mlx-src'
assert (new_source/'a.txt').read_text() == 'after\n'
assert (new_source/'.pmetal-patch-series-v1').exists()
assert (fixture/'build/_deps/pmetal-aabbcc/mlx-build').is_dir()
assert snapshot(old_source) == old_before, 'upgrade changed old patched stage'
assert snapshot(legacy) == legacy_before, 'upgrade changed reference legacy native source'
print('PASS: real local-only CMake FetchContent upgrade preserves old patched/no-receipt stage and creates new content-keyed source+binary stage', flush=True)
print('PASS: patch source/receipt verification and upgrade tests; no native runtime claim', flush=True)
