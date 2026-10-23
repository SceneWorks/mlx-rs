#!/bin/bash
# Hosted/nax only: compile and execute focused actual CPU+Metal native tests.
# This is invoked by validate.yml's existing stable job, never on the dev Mac.
set -euo pipefail
root=$(pwd -P)
sources=()
outer_builds=()
for manifest in target/debug/build/pmetal-mlx-sys-*/out/build/lib/pmetal-mlx-prebuilt.txt; do
  [ -f "$manifest" ] || continue
  key=$(sed -n 's/^fingerprint=//p' "$manifest")
  outer=${manifest%/lib/pmetal-mlx-prebuilt.txt}
  source="$root/$outer/_deps/pmetal-$key/mlx-src"
  [ -f "$source/.pmetal-patch-series-v1" ] || continue
  sources+=("$source")
  outer_builds+=("$root/$outer")
done
[ "${#sources[@]}" -eq 1 ] || { printf 'expected one current receipted native source, found %s\n' "${#sources[@]}" >&2; exit 1; }
source=${sources[0]}
outer=${outer_builds[0]}
staged=${outer%/build}/mlx-c-staged
# Reverify exact ordered inputs and final native bytes; marker presence is not evidence.
(cd "$source" && sh "$staged/patches/apply_patches.sh")

python3 -I mlx-sys/tests/run_completion_retirement.py --native-source "$source" --work-dir "$RUNNER_TEMP/metal-owner-state"
python3 -I mlx-sys/tests/run_cpu_retirement.py --native-source "$source" --work-dir "$RUNNER_TEMP/cpu-owner-state"

build="$root/target/native-retirement-tests"
args=()
# Reuse existing fetched header-only dependencies; doctest is fetched only here.
for pair in METAL_CPP:metal_cpp FMT:fmt JSON:json GGUFLIB:gguflib; do
  variable=${pair%%:*}
  name=${pair#*:}
  [ ! -d "$outer/_deps/$name-src" ] || args+=("-DFETCHCONTENT_SOURCE_DIR_$variable=$outer/_deps/$name-src")
done
cmake -S "$source" -B "$build" \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}" \
  -DMLX_BUILD_METAL=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF "${args[@]}"
cmake --build "$build" --target completion_retirement_tests --parallel 4
# Separate processes exercise Event and fast FenceImpl ownership paths.
# Exactly 20 Debug test cases per process: actual callback throws, per-array
# failures, same-stream recovery, immutable/observed receipt acknowledgement,
# gated distinct copies, idle identity and post-flush callback capture. The mixed
# graphs also ensure CPU workers never block on GPU buffer-outcome futures.
for fast in 0 1; do
  DEVICE=cpu MLX_METAL_FAST_SYNCH="$fast" "$build/tests/completion_retirement_tests" \
    --test-case='*completion retirement*' --no-skip
 done
