extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf, process::Command};

/// The clang runtime variant to link, chosen for the **target**, not the host.
///
/// This link exists solely as a macOS 26+ workaround (see the call site): Xcode's clang runtime
/// supplies `___isPlatformVersionAtLeast` when the bundled LLVM runtime is outdated. It is
/// **macOS-specific**, so every other target gets `None`:
///
/// - `libclang_rt.osx.a` linked into an iOS binary is rejected with "Unsupported archive
///   identifier", which is what made cross-compilation fail before this was target-aware;
/// - the `ios` / `iossim` archives are rejected the same way *and* are not needed — the Rust
///   toolchain supplies the equivalent on those targets.
///
/// Build scripts are compiled for the host, so `cfg!(target_os = ...)` in this file always reports
/// the *host* platform when cross-compiling. Read cargo's target environment instead.
fn clang_rt_variant() -> Option<&'static str> {
    match env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "macos" => Some("osx"),
        _ => None,
    }
}

/// Find the clang runtime library path dynamically using xcrun.
///
/// `variant` is the platform suffix from [`clang_rt_variant`] — e.g. `osx`, `ios`, `iossim`.
fn find_clang_rt_path(variant: &str) -> Option<String> {
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
        let clang_rt_lib = darwin_path.join(format!("libclang_rt.{variant}.a"));
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

/// The Apple platform being built for, resolved from the **target**, not the host.
///
/// Each variant carries the environment variable Apple's toolchain reads for that platform, the
/// CMake system name (`None` for macOS — a native build sets no `CMAKE_SYSTEM_NAME`), and MLX's
/// minimum version for Metal support.
struct ApplePlatform {
    deployment_env: &'static str,
    cmake_system_name: Option<&'static str>,
    min_version: (u32, u32),
}

/// Resolve the Apple platform from cargo's target environment.
///
/// Build scripts compile for the host, so `cfg!(target_os = ...)` here reports the *host* even
/// when cross-compiling. `CARGO_CFG_TARGET_OS` is the target.
fn apple_platform() -> Option<ApplePlatform> {
    match env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        // MLX requires macOS >= 14.0 for Metal. Cargo/Tauri often default to 10.13, which MLX's
        // CMakeLists.txt rejects.
        "macos" => Some(ApplePlatform {
            deployment_env: "MACOSX_DEPLOYMENT_TARGET",
            cmake_system_name: None,
            min_version: (14, 0),
        }),
        // iOS 18.0, chosen to match Metal versions rather than picked for recency:
        //   iOS 16 -> Metal 300, iOS 17 -> Metal 310, iOS 18 -> Metal 320.
        // MLX's own macOS floor (14.0) is Metal 310, so iOS 16 would be *below* the baseline
        // its kernels assume. 18.0 also keeps `fence` coherent: the kernel is only built when
        // MLX_METAL_VERSION >= 320, while fence.cpp's runtime guard is
        // `__builtin_available(macOS 15, iOS 18, *)`. Building at iOS 17 would satisfy the
        // runtime check on an iOS 18 device while the kernel was never compiled in — a missing
        // kernel at `get_kernel("fence_wait")`. Latent today (MLX_METAL_FAST_SYNCH defaults
        // off) but a real trap.
        // Note this is a *different* knob from MACOSX_DEPLOYMENT_TARGET — leaking the macOS
        // value in here (e.g. "26.2") silently sets an iOS floor that excludes every shipping
        // device.
        "ios" => Some(ApplePlatform {
            deployment_env: "IPHONEOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("iOS"),
            min_version: (18, 0),
        }),
        _ => None,
    }
}

/// Resolve the deployment target for `platform`, honoring a higher explicitly-set value.
fn resolve_deployment_target(platform: &ApplePlatform) -> String {
    let (min_major, min_minor) = platform.min_version;

    if let Ok(val) = env::var(platform.deployment_env) {
        let parts: Vec<u32> = val.split('.').filter_map(|s| s.parse().ok()).collect();
        let major = parts.first().copied().unwrap_or(0);
        let minor = parts.get(1).copied().unwrap_or(0);
        if (major, minor) >= (min_major, min_minor) {
            return val;
        }
    }
    format!("{min_major}.{min_minor}")
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
    //     retired as redundant. This patch is regenerated to add ONLY the debug-only
    //     (NDEBUG-gated) test hook mlx_pmetal_test_inject_command_buffer_error, which
    //     lets mlx-tests/tests/command_buffer_recoverable.rs drive that upstream
    //     recovery path deterministically (without tripping the real GPU watchdog).
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
    //   - ios-metal-sdk.patch                 : cross-compile the Metal kernels for iOS.
    //     MLX's kernel rules hardcode `xcrun -sdk macosx metal` and
    //     `-mmacosx-version-min`, so a CMAKE_SYSTEM_NAME=iOS build still emitted a
    //     **macOS** metallib: the C++ cross-compiled, the binary linked, and the
    //     artifact would have failed on device. (MLX's root CMakeLists already
    //     guards its SDK probe on `CMAKE_SYSTEM_NAME STREQUAL "Darwin"` and falls
    //     back to MLX_METAL_VERSION 0, so upstream anticipated non-macOS
    //     configuration — the kernel rules just never got the matching branch.)
    //     The patch selects the SDK and version-min flag by platform in both the
    //     per-kernel compile and the metallib link, and adds an iOS arm to the root
    //     probe so MLX_METAL_VERSION is detected rather than pinned to 0 (0 would
    //     silently drop every MLX_METAL_VERSION-gated kernel).
    //     Deployment target matters here: iOS 16 -> Metal 300, 17 -> 310, 18 -> 320.
    //     MLX's macOS floor (14.0) is Metal 310, so iOS 16 is BELOW the baseline its
    //     kernels assume; resolve_deployment_target() floors iOS at 18.0 (Metal 320).
    //     That also keeps `fence` coherent — the kernel is built only at
    //     MLX_METAL_VERSION >= 320 while fence.cpp's runtime guard is
    //     `__builtin_available(macOS 15, iOS 18, *)`, so a lower floor could satisfy
    //     the runtime check with the kernel absent.
    //     The NAX gate fails safe on iOS: it needs Metal >= 400 AND
    //     MACOS_SDK_VERSION >= 26.2, and the latter is unset off macOS.
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
    // sc-12780: ALL THREE patches are now `required = true`. The metallib and command-buffer
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
        ("patches/ios-metal-sdk.patch", true),
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
    // Resolve and export the deployment target early so the cmake crate (and the cc crate) don't
    // inject a lower -m<platform>-version-min into CFLAGS/CXXFLAGS. Without this, cargo's default
    // (10.13 on macOS, 10.0 on iOS) causes MLX's CMakeLists.txt to reject the build — and on iOS
    // an unset target also drops __chkstk_darwin (libSystem, iOS 12+) at link time.
    let platform = apple_platform();
    if let Some(platform) = &platform {
        let target = resolve_deployment_target(platform);
        env::set_var(platform.deployment_env, &target);
    }

    let mlx_c_src = prepare_mlx_c_source();
    let mut config = Config::new(&mlx_c_src);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // mlx-c's example .app targets are never linked into this crate, and they do not build for
    // every Apple platform: on the iOS **simulator** they fail with undefined `_MTLIOErrorDomain`
    // (MTLIO is absent from the simulator's Metal framework), which fails the whole cmake build
    // even though libmlx/libmlxc themselves compiled cleanly. Skipping them is free everywhere and
    // unblocks aarch64-apple-ios-sim.
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");

    if let Some(platform) = &platform {
        let target = resolve_deployment_target(platform);
        // CMAKE_OSX_DEPLOYMENT_TARGET is the knob for every Apple platform, not just macOS.
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &target);
        if let Some(system_name) = platform.cmake_system_name {
            // Cross-compiling: tell CMake the target platform and architecture explicitly.
            config.define("CMAKE_SYSTEM_NAME", system_name);
            if let Ok(arch) = env::var("CARGO_CFG_TARGET_ARCH") {
                let arch = if arch == "aarch64" { "arm64" } else { &arch };
                config.define("CMAKE_OSX_ARCHITECTURES", arch);
            }
        }
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
    if let Some(clang_rt) = clang_rt_variant() {
        if let Some(clang_rt_path) = find_clang_rt_path(clang_rt) {
            println!("cargo:rustc-link-search={}", clang_rt_path);
            println!("cargo:rustc-link-lib=static=clang_rt.{clang_rt}");
        }
    }

    // Publish the built metallib's path to dependents as DEP_MLX_METALLIB (requires the `links`
    // key in Cargo.toml). This is the supported way off the host: `~/.cache/pmetal/lib` does not
    // exist inside an iOS app sandbox, and the compiled-in METAL_PATH points into the cargo
    // target directory, which is not shipped. An iOS packaging step reads this and copies the
    // metallib into the .app, where MLX's `load_colocated_library` finds it next to the
    // executable. Emitted on every platform — it is just a path — so a macOS packager can use it
    // too rather than reaching into target/ by hand.
    #[cfg(feature = "metal")]
    {
        let metallib = dst.join("build/lib/mlx.metallib");
        if metallib.exists() {
            println!("cargo:metallib={}", metallib.display());
        }
    }

    // Cache mlx.metallib to ~/.cache/pmetal/lib/ so the binary works regardless
    // of where it's installed. This is critical for `cargo install` where the
    // build directory is cleaned up after the binary is placed.
    //
    // macOS ONLY. The cache is a single shared, platform-agnostic path, so a cross-compiled
    // build would silently overwrite the host's metallib with kernels for another platform —
    // and the resolver has no way to tell them apart. Local `cargo test`/`run` binaries have no
    // compiled-in metallib and resolve *solely* through this cache, so poisoning it breaks every
    // subsequent macOS test run until a native build repairs it. An iOS app bundles its metallib
    // in the `.app` and never reads `$HOME` anyway (there is nothing useful there in the sandbox).
    #[cfg(feature = "metal")]
    if matches!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("macos") | Err(_)
    ) {
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
