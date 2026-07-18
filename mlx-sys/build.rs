extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf, process::Command};

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

/// Resolve the macOS deployment target.
///
/// Enforces a minimum of 14.0 (MLX's requirement for Metal support).
/// If `MACOSX_DEPLOYMENT_TARGET` is set to a higher value, that is used instead.
/// Cargo/Tauri often default to 10.13, which MLX's CMakeLists.txt rejects.
#[cfg(target_os = "macos")]
fn resolve_deployment_target() -> String {
    const MLX_MIN_MACOS: (u32, u32) = (14, 0);

    if let Ok(val) = env::var("MACOSX_DEPLOYMENT_TARGET") {
        let parts: Vec<u32> = val.split('.').filter_map(|s| s.parse().ok()).collect();
        let major = parts.first().copied().unwrap_or(0);
        let minor = parts.get(1).copied().unwrap_or(0);
        if (major, minor) >= MLX_MIN_MACOS {
            return val;
        }
    }
    format!("{}.{}", MLX_MIN_MACOS.0, MLX_MIN_MACOS.1)
}

/// Copy src/mlx-c to a staging directory and inject the metallib search-path
/// patch into the CMakeLists.txt. This avoids modifying the mlx-c git submodule
/// while ensuring the patch is applied when MLX is fetched via FetchContent.
fn prepare_mlx_c_source() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let staged = out_dir.join("mlx-c-staged");
    let src = PathBuf::from("src/mlx-c");

    // Copy the entire mlx-c source tree to the staging area
    if staged.exists() {
        std::fs::remove_dir_all(&staged).expect("Failed to clean staged mlx-c");
    }
    copy_dir_recursive(&src, &staged).expect("Failed to copy mlx-c to staging");

    // sc-2781: patch the STAGED mlx-c fft bindings so mlx-c 0.6.0 compiles against MLX 0.31.2.
    // 0.31.2 inserted an `FFTNorm norm = FFTNorm::Backward` parameter before `StreamOrDevice s`
    // across the fftn/ifftn/fft/ifft/rfft/irfft family; mlx-c 0.6.0 calls `fft*(... stream)`, so
    // `stream` mis-binds to `norm` and won't compile. The patch passes `FFTNorm::Backward`
    // explicitly at the 12 norm call sites (the 2 shift fns are unchanged), keeping the C API
    // identical → no Rust-binding cascade. The submodule stays pristine at Apple upstream v0.6.0.
    // TEMPORARY: drop once Apple tags an mlx-c release that supports 0.31.2 (main already does).
    // Unlike the MLX combined.patch (applied to the fetched MLX git checkout via CMake
    // PATCH_COMMAND), this targets the copied mlx-c source tree, so we apply it here with `patch`.
    //
    // sc-12745 (MLX 0.31.2 -> 0.32.0) CARRY-FORWARD: this patch is NOT 0.31.2-specific and is
    // NOT droppable at 0.32.0. sc-12744 measurement confirmed 0.32.0's mlx/fft.h STILL declares
    // `FFTNorm norm = FFTNorm::Backward` before `StreamOrDevice s`, while Apple's pinned mlx-c
    // 0.6.0 still calls `fft*(... stream)` — dropping the patch yields exactly 12 "no matching
    // function" errors across the fft/ifft/rfft/irfft family. The patch targets the unchanged
    // pinned mlx-c 0.6.0 source, so it applies cleanly to 0.32.0 as-is.
    let fft_patch =
        std::fs::canonicalize("patches/mlx-c-fft-norm.patch").expect("find mlx-c-fft-norm.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&fft_patch)
        .status()
        .expect("Failed to run `patch` for mlx-c-fft-norm.patch");
    assert!(
        status.success(),
        "mlx-c-fft-norm.patch failed to apply to staged mlx-c (sc-2781)"
    );
    println!("cargo:rerun-if-changed=patches/mlx-c-fft-norm.patch");

    // Copy our patch files into the staged source. FetchContent allows only one
    // PATCH_COMMAND, so build.rs generates apply_patches.sh (below) which applies
    // each MLX source patch individually and idempotently.
    //   - metallib-search-path.patch         : pmetal metallib resolver (device.cpp
    //     load_default_library). Adds a PMETAL_METALLIB_PATH env override, a
    //     ~/.cache/pmetal/lib/mlx.metallib user-cache lookup, and (sc-7898) prefers
    //     THIS build's colocated metallib over the shared user cache so a stale
    //     cached metallib can never shadow the current build's kernels. sc-12780:
    //     regenerated for MLX 0.32.0 — rebased onto the refactored resolver, which
    //     now also carries upstream metal::set_metallib_path() (#3597); that single
    //     explicit override is preserved (inert unless called) and does NOT provide
    //     the env/user-cache/colocated-precedence behavior, so this patch is still
    //     required.
    //   - command-buffer-recoverable.patch   : sc-5009 — a recoverable Metal
    //     command-buffer error (kIOGPUCommandBufferCallbackErrorTimeout / OOM)
    //     reported from a completion handler on an internal Metal thread, where
    //     throwing would `std::terminate`. sc-12780: MLX 0.32.0 implements this
    //     recovery UPSTREAM (CommandEncoder::commit records the error into the
    //     per-encoder error_ / poisons signal events instead of throwing, and it is
    //     re-thrown synchronously on the waiting thread via synchronize() /
    //     EventImpl::check_error). So the 0.31.2 production record-not-throw logic is
    //     retired as redundant. The patch adds ONLY the debug-only (NDEBUG-gated)
    //     test hook mlx_pmetal_test_inject_command_buffer_error, which lets
    //     mlx-tests/tests/command_buffer_recoverable.rs drive that upstream recovery
    //     path deterministically (without tripping the real GPU watchdog). sc-12786:
    //     the hook now injects THROUGH the real upstream mechanism — Event::signal
    //     drains it on the host thread and poisons the event via the real
    //     EventImpl::set_error, so the error is re-raised by unpatched upstream code
    //     (EventImpl::wait -> check_error). Previously it was drained by a patched-in
    //     check in the outer Event::wait BEFORE EventImpl::wait, which could not
    //     catch a regression of the real set_error/check_error plumbing.
    //   - pad-copy-int64.patch                : sc-12746 (epic 12742) — fix the
    //     copy_gpu_inplace dispatch gate in backend/metal/copy.cpp. pad() and
    //     concatenate() hand copy_gpu_inplace a slice-VIEW `out` whose
    //     data_size() is only the small copied input-region while its strides
    //     are the FULL destination's, so the `out.data_size() > INT32_MAX` gate
    //     under-reports the addressable destination span, picks the int32 `gg`
    //     copy kernel, and its int32 dst_idx overflows above 2^31 -> silent
    //     corruption of large (>i32::MAX-element) pad/concat outputs. The patch
    //     bases `large` on the true reachable element span
    //     (offset + Σ (shape[i]-1)·|stride|) for both sides, routing pad/concat
    //     onto the EXISTING int64 (`*large`) kernels — no copy.metal edit. This
    //     is exactly the follow-up MLX PR #3524 (0.32.0) deferred for pad/concat;
    //     an upstream ml-explore/mlx PR is prepared separately.
    //   - thread-shared-streams.patch         : sc-12937 — MLX registers each
    //     stream's command encoder in a thread_local map, so a stream is only
    //     usable from the thread that created it and any cross-thread eval
    //     throws "There is no Stream(gpu, N) in current thread." (metal
    //     device.cpp get_command_encoder; same for cpu). Rust mlx-rs arrays
    //     are Send — the test harness runs every #[test] on its own thread,
    //     pmetal moves work across tokio workers, and MLX's global random key
    //     state crosses threads by design — so gpu::new_stream/cpu::new_stream
    //     are redirected to the global-map registration (upstream's own
    //     new_thread_unsafe_stream, #3578), restoring cross-thread stream
    //     visibility. Concurrent eval on the SAME stream from two threads
    //     remains the caller's responsibility (unchanged from upstream).
    //   - thread-safe-eval.patch             : sc-12959 — closes the three
    //     hazards the sc-12937 patch left open, making the fork's threading
    //     model coherent ("streams visible everywhere, all eval serialized,
    //     O(devices) streams"):
    //     (1) process-global recursive eval lock (detail::eval_mutex) taken by
    //         eval_impl and synchronize — concurrent cross-thread eval now
    //         serializes instead of racing a shared command encoder into a
    //         Metal SIGABRT. eval()'s host-side completion wait stays OUTSIDE
    //         the lock (synchronize(Stream) drains under it by design). This
    //         enforces the pmetal consumer contract (single eval thread,
    //         batch-dimension concurrency) that was previously discipline-only.
    //     (2) shared_mutex on the global encoder maps (metal + cpu) — the
    //         fallback read in get_command_encoder no longer races another
    //         thread's first-stream registration (unordered_map rehash UB).
    //     (3) process-global default stream (was per-thread) — a dying worker
    //         thread (tokio spawn_blocking churn) no longer strands one
    //         CommandEncoder + MTLCommandQueue per thread in the global map;
    //         stream count is O(devices). Safe because of (1).
    //
    // The sc-2714 (dense GEMM) and sc-2770 (fast SDPA) NAX-dispatch gate patches were
    // REMOVED in sc-2772. Their root cause was never the dispatch or the MLX version: the
    // NAX matrix-unit kernels (`mpp::tensor_ops::matmul2d`) are only valid for macOS >= 26.2
    // (the `is_nax_available()` floor), but the metal kernels were being compiled with
    // `-mmacosx-version-min=26.0` (the old MACOSX_DEPLOYMENT_TARGET), BELOW that floor, so
    // metalfe miscompiled the tensor-op intrinsic to garbage for 16-bit. Compiling the
    // kernels at >= 26.2 (mlx-gen's .cargo/config.toml now sets 26.2) makes the NAX 16-bit
    // GEMM + SDPA correct, so the dispatch gates are unnecessary and 16-bit now uses the
    // (correct, faster) NAX matrix unit. The `bf16_matmul_sweep` + `sdpa_nax_repro` tripwires
    // still assert 16-bit correctness — now they guard the deployment-target fix.
    let patches_dir = staged.join("patches");
    std::fs::create_dir_all(&patches_dir).expect("Failed to create patches dir");
    // (basename, required). `required = true` means the build MUST fail if the patch
    // does not apply; `false` means best-effort (a fully-failing patch is a safe no-op).
    //
    // sc-12746: the MLX source patches are applied INDIVIDUALLY, not concatenated into
    // one `combined.patch`. `git apply` is atomic PER INVOCATION, so a single combined patch
    // meant ANY stale hunk silently dropped ALL patches (the previous `git apply combined.patch
    // || true` swallowed the failure). Applying each patch on its own keeps a genuine failure
    // isolated to its own patch, and per-file atomicity prevents a dangerous half-apply of a
    // partially-matching patch.
    //
    // sc-12780: ALL patches are `required = true` (three at the time; sc-12937
    // adds a fourth under the same policy). The metallib and command-buffer
    // patches were regenerated for MLX 0.32.0 (their 0.31.2 context had drifted and both were
    // temporarily demoted to best-effort no-ops by sc-12746, which is exactly how the sc-12745
    // bump silently shipped WITHOUT them). Re-arming them to required=true makes a future silent
    // drop impossible: if a regenerated patch ever fails to apply again, the build aborts loudly
    // instead of producing a binary missing pmetal's metallib resolver or the recoverable-error
    // test hook. See the per-patch notes above and the idempotency guard below.
    let patch_files = [
        ("patches/metallib-search-path.patch", true),
        ("patches/command-buffer-recoverable.patch", true),
        ("patches/pad-copy-int64.patch", true),
        ("patches/thread-shared-streams.patch", true),
        ("patches/thread-safe-eval.patch", true),
    ];
    // sc-12780 idempotency guard: CMake FetchContent may re-run PATCH_COMMAND against an
    // mlx-src that is ALREADY patched (e.g. an incremental rebuild that does not re-fetch).
    // `git apply` is not idempotent — re-applying an applied patch fails "patch does not apply".
    // So each patch is guarded: if it reverse-applies cleanly it is already present and we SKIP
    // (no-op); otherwise we apply it forward. A `required` patch that genuinely cannot apply
    // (neither already-present nor forward-applicable) aborts the build. This keeps re-runs a
    // no-op while still hard-failing on a real apply failure — the whole point of required=true.
    let mut script = String::from(
        "#!/bin/sh\n\
         # Generated by mlx-sys/build.rs. Apply each MLX source patch individually,\n\
         # idempotently (sc-12780). cwd is the fetched mlx-src (a git repo); patches\n\
         # live next to this script.\n\
         d=\"$(dirname \"$0\")\"\n",
    );
    for (pf, required) in patch_files {
        let name = std::path::Path::new(pf)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        std::fs::copy(pf, patches_dir.join(&name))
            .unwrap_or_else(|e| panic!("Failed to copy {pf}: {e}"));
        if required {
            script.push_str(&format!(
                "if git apply --reverse --check \"$d/{name}\" 2>/dev/null; then\n  \
                 echo \"pmetal: {name} already applied; skipping\" >&2\n\
                 elif git apply \"$d/{name}\"; then\n  :\n\
                 else\n  \
                 echo \"pmetal: FATAL required patch {name} (sc-12780) failed to apply\" >&2\n  \
                 exit 1\n\
                 fi\n"
            ));
        } else {
            script.push_str(&format!(
                "if git apply --reverse --check \"$d/{name}\" 2>/dev/null; then\n  \
                 echo \"pmetal: {name} already applied; skipping\" >&2\n\
                 elif git apply \"$d/{name}\" 2>/dev/null; then\n  :\n\
                 else\n  \
                 echo \"pmetal: {name} did not apply (best-effort, skipped)\" >&2\n\
                 fi\n"
            ));
        }
    }
    let apply_script = patches_dir.join("apply_patches.sh");
    std::fs::write(&apply_script, &script).expect("Failed to write apply_patches.sh");

    // Inject PATCH_COMMAND into the FetchContent_Declare for MLX, and bump the fetched MLX tag
    // v0.31.1 -> v0.32.0 (sc-2781 pinned 0.31.2 for byte-parity; sc-12745 bumps to 0.32.0 —
    // measured + audited GO in slice 2). PATCH_COMMAND now runs apply_patches.sh (per-file
    // application, sc-12746) instead of a single `git apply combined.patch`.
    let cmake_path = staged.join("CMakeLists.txt");
    let cmake_content =
        std::fs::read_to_string(&cmake_path).expect("Failed to read CMakeLists.txt");
    let patched = cmake_content.replace(
        "GIT_TAG v0.31.1)",
        "GIT_TAG v0.32.0\n    PATCH_COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/patches/apply_patches.sh)",
    );
    std::fs::write(&cmake_path, patched).expect("Failed to write patched CMakeLists.txt");

    // Tell cargo to rerun if any patch changes
    for (pf, _required) in patch_files {
        println!("cargo:rerun-if-changed={pf}");
    }

    staged
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else if ty.is_symlink() {
            // Resolve symlinks (common in git submodules)
            let target = std::fs::read_link(entry.path())?;
            let resolved = if target.is_absolute() {
                target
            } else {
                entry.path().parent().unwrap().join(&target)
            };
            if resolved.is_dir() {
                copy_dir_recursive(&resolved, &dest_path)?;
            } else {
                std::fs::copy(&resolved, &dest_path)?;
            }
        } else {
            std::fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

fn build_and_link_mlx_c() {
    // MLX requires macOS >= 14.0 for Metal support. Override the deployment
    // target early so the cmake crate (and cc crate) don't inject a lower
    // -mmacosx-version-min flag into CFLAGS/CXXFLAGS. Without this, Cargo's
    // default target (10.13) causes MLX's CMakeLists.txt to reject the build.
    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        env::set_var("MACOSX_DEPLOYMENT_TARGET", &target);
    }

    let mlx_c_src = prepare_mlx_c_source();
    let mut config = Config::new(&mlx_c_src);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &target);
    }

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }

    // Cache mlx.metallib to ~/.cache/pmetal/lib/ so the binary works regardless
    // of where it's installed. This is critical for `cargo install` where the
    // build directory is cleaned up after the binary is placed.
    #[cfg(feature = "metal")]
    {
        let metallib = dst.join("build/lib/mlx.metallib");
        if metallib.exists() {
            if let Ok(home) = env::var("HOME") {
                let cache_dir = PathBuf::from(home).join(".cache/pmetal/lib");
                let dest = cache_dir.join("mlx.metallib");
                let should_copy = if dest.exists() {
                    // Replace if the build artifact is newer
                    dest.metadata()
                        .and_then(|d| {
                            metallib.metadata().map(|s| {
                                s.modified()
                                    .ok()
                                    .zip(d.modified().ok())
                                    .is_some_and(|(src_t, dst_t)| src_t > dst_t)
                            })
                        })
                        .unwrap_or(false)
                } else {
                    true
                };
                if should_copy {
                    let _ = std::fs::create_dir_all(&cache_dir);
                    match std::fs::copy(&metallib, &dest) {
                        Ok(_) => {
                            println!("cargo:warning=Cached mlx.metallib to {}", dest.display())
                        }
                        Err(e) => println!("cargo:warning=Failed to cache mlx.metallib: {}", e),
                    }
                }
            }
        }
    }
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
