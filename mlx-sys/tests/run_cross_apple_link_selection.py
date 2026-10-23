#!/usr/bin/env python3
"""CPU-only filesystem regression for cross-apple's actual manifest-keyed link selection."""
import argparse
from pathlib import Path
import shutil
import subprocess
import textwrap

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, required=True)
args = parser.parse_args()
args.work_dir.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
workflow = (root / '.github/workflows/cross-apple.yml').read_text()
start = workflow.index('          # Select the unique completed Debug output')
end = workflow.index('          sdk=$(xcrun', start)
selection = textwrap.dedent(workflow[start:end])
assert '"$gguf" $shim' in workflow, 'link must use the selected archive'
assert '"$out/build/_deps/mlx-build/mlx/io/libgguflib.a"' not in workflow

target = 'aarch64-apple-ios'
key_a, key_b = 'a' * 64, 'b' * 64
script = args.work_dir / 'select.sh'
script.write_text('set -euo pipefail\n' + selection.replace('${{ matrix.target }}', target)
                  + 'printf "%s\\n" "$out" "$gguf"\n')
subprocess.run(['bash', '-n', str(script)], check=True)
relative = Path('target') / target / 'debug/build/pmetal-mlx-sys-current/out'
manifest_name = 'build/lib/pmetal-mlx-prebuilt.txt'

def fixture(name):
    case = args.work_dir / name
    out = case / relative
    fields = {'schema': '1', 'target': target, 'build_type': 'Debug',
              'deployment_target': '18.0', 'features': 'accelerate,metal', 'fingerprint': key_a}
    for file in ['mlx-c-staged/examples/example.c', 'build/lib/libmlx.a', 'build/lib/libmlxc.a',
                 f'build/_deps/pmetal-{key_a}/mlx-build/mlx/io/libgguflib.a',
                 f'build/_deps/pmetal-{key_b}/mlx-build/mlx/io/libgguflib.a',
                 'build/_deps/mlx-build/mlx/io/libgguflib.a']:
        p = out / file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file)
    save(out, fields)
    return case, out, fields

def save(out, fields):
    (out / manifest_name).write_text(''.join(f'{key}={value}\n' for key, value in fields.items()))

def run(case, good, key=key_a):
    result = subprocess.run(['bash', str(script)], cwd=case, capture_output=True, text=True)
    if good:
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.splitlines() == [str(relative), str(relative / f'build/_deps/pmetal-{key}/mlx-build/mlx/io/libgguflib.a')]
    else:
        assert result.returncode != 0, 'invalid selection unexpectedly succeeded: ' + str(case)
    print('PASS: ' + case.name, flush=True)

case, out, fields = fixture('matching-key-not-other-key-or-legacy')
run(case, True)
case, out, fields = fixture('manifest-key-change-selects-only-that-key')
fields['fingerprint'] = key_b
save(out, fields)
run(case, True, key_b)
case, out, fields = fixture('missing-matching-archive-rejects-despite-other-key-and-legacy')
(out / f'build/_deps/pmetal-{key_a}/mlx-build/mlx/io/libgguflib.a').unlink()
run(case, False)
for field, value in [('target', 'aarch64-apple-ios-sim'), ('build_type', 'Release'),
                     ('deployment_target', '17.0'), ('features', 'metal'),
                     ('schema', '2'), ('fingerprint', '../not-a-fingerprint')]:
    case, out, fields = fixture('reject-wrong-' + field)
    fields[field] = value
    save(out, fields)
    run(case, False)
case, out, fields = fixture('reject-no-completed-manifest')
(out / manifest_name).unlink()
run(case, False)
case, out, fields = fixture('reject-ambiguous-completed-outputs')
shutil.copytree(out, out.parent.parent / 'pmetal-mlx-sys-other/out')
run(case, False)
print('PASS: 11 actual workflow selection cases; no native compiler/link/GPU/mock commands', flush=True)
