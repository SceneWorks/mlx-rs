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
/// `variant` is the platform suffix from [`clang_rt_variant`], which only ever yields `osx` —
/// every other Apple platform deliberately links no clang runtime (see that function). The
/// parameter exists so the archive name and the platform decision stay in one place.
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
struct ApplePlatform {
    /// The environment variable Apple's toolchain reads for this platform's deployment target.
    deployment_env: &'static str,
    /// `CMAKE_SYSTEM_NAME`. `None` for macOS — a native build sets none.
    cmake_system_name: Option<&'static str>,
    /// The `xcrun -sdk` name, also used to resolve `CMAKE_OSX_SYSROOT`.
    ///
    /// Device and simulator are *different SDKs*, and the arch does not distinguish them: an
    /// Apple-silicon simulator is `arm64`, exactly like the device. Selecting `iphoneos` for a
    /// simulator build produces `air64-apple-ios` kernels that fail at `newLibraryWithURL`.
    sdk: &'static str,
    /// The `metal` version-min flag prefix matching [`Self::sdk`], or `None` on platforms with
    /// no Metal framework at all (watchOS). Simulators take a *different* flag from their
    /// device counterpart (`-mios-simulator-version-min=` vs `-mios-version-min=`).
    metal_version_min_prefix: Option<&'static str>,
    /// The minimum OS version this build targets.
    min_version: (u32, u32),
}

/// Resolve the Apple platform from cargo's target environment.
///
/// Build scripts compile for the host, so `cfg!(target_os = ...)` here reports the *host* even
/// when cross-compiling. `CARGO_CFG_TARGET_OS` is the target.
///
/// Returns `None` for non-Apple targets and **panics** for an Apple target with no arm below.
/// The panic is deliberate: falling through to `None` leaves `CMAKE_SYSTEM_NAME` to cmake-rs,
/// which sets e.g. `visionOS` on its own, and MLX's root probe then pins `MLX_METAL_VERSION` to
/// 0 — silently dropping every version-gated kernel and producing a metallib that looks fine and
/// is missing most of its kernels. A loud failure at configure time beats that.
fn apple_platform() -> Option<ApplePlatform> {
    // MLX's Metal floors are stated as Metal *versions*, so each platform's minimum is the OS
    // that first ships the required Metal version rather than a number picked for recency
    // (measured with `echo __METAL_VERSION__ | xcrun -sdk <sdk> metal -m<plat>-version-min=<v>
    // -E -x metal -P -`, Xcode 26.5):
    //
    //           Metal 300   Metal 310   Metal 320
    //   macOS       13          14          15
    //   iOS         16          17          18
    //   tvOS        16          17          18
    //
    // MLX's own macOS floor (14.0) is Metal 310, so anything below the 310 row is *under* the
    // baseline its kernels assume. 320 additionally keeps `fence` coherent: the kernel is only
    // built when MLX_METAL_VERSION >= 320, while fence.cpp's runtime guard is
    // `__builtin_available(macOS 15, iOS 18, *)`. Building at iOS 17 would satisfy the runtime
    // check on an iOS 18 device while the kernel was never compiled in — a missing kernel at
    // `get_kernel("fence_wait")`. Latent today (MLX_METAL_FAST_SYNCH defaults off) but a real
    // trap, and worse on tvOS, where that guard's `*` arm makes the runtime check pass
    // unconditionally.
    //
    // Each platform reads its *own* deployment-target variable. Leaking the macOS value into
    // one of the others (e.g. "26.2") silently sets a floor that excludes every shipping device.
    //
    // Excluding iOS/tvOS 16 and 17 is a deliberate product decision, not a consequence of the
    // Metal reasoning above: shipping iOS is 26.x, so an 18.0 floor is two majors back. Note
    // this is stricter than mlx-swift upstream, which targets iOS 16. Lowering it to 17.0 would
    // still satisfy MLX's Metal-310 baseline but reintroduces the `fence` hazard; 16.0 is below
    // the baseline outright and apple-metal-sdk.patch will reject it at configure time.
    const METAL_320: (u32, u32) = (18, 0);

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // `sim` is how rustc spells "this Apple target runs in a simulator": aarch64-apple-ios-sim,
    // x86_64-apple-ios, and the -sim variants for tvOS/watchOS all set it. Nothing else
    // distinguishes a simulator target — in particular the arch does not.
    let sim = env::var("CARGO_CFG_TARGET_ABI").as_deref() == Ok("sim");

    match (os.as_str(), sim) {
        // MLX requires macOS >= 14.0 for Metal. Cargo/Tauri often default to 10.13, which MLX's
        // CMakeLists.txt rejects.
        ("macos", _) => Some(ApplePlatform {
            deployment_env: "MACOSX_DEPLOYMENT_TARGET",
            cmake_system_name: None,
            sdk: "macosx",
            metal_version_min_prefix: Some("-mmacosx-version-min="),
            min_version: (14, 0),
        }),
        ("ios", false) => Some(ApplePlatform {
            deployment_env: "IPHONEOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("iOS"),
            sdk: "iphoneos",
            metal_version_min_prefix: Some("-mios-version-min="),
            min_version: METAL_320,
        }),
        ("ios", true) => Some(ApplePlatform {
            deployment_env: "IPHONEOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("iOS"),
            sdk: "iphonesimulator",
            metal_version_min_prefix: Some("-mios-simulator-version-min="),
            min_version: METAL_320,
        }),
        ("tvos", false) => Some(ApplePlatform {
            deployment_env: "TVOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("tvOS"),
            sdk: "appletvos",
            metal_version_min_prefix: Some("-mtvos-version-min="),
            min_version: METAL_320,
        }),
        ("tvos", true) => Some(ApplePlatform {
            deployment_env: "TVOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("tvOS"),
            sdk: "appletvsimulator",
            metal_version_min_prefix: Some("-mtvos-simulator-version-min="),
            min_version: METAL_320,
        }),
        // watchOS ships **no Metal framework** — it is absent from both the WatchOS and
        // WatchSimulator SDKs, so no deployment target makes MLX_BUILD_METAL work here.
        // `metal_version_min_prefix: None` records that, and build_and_link_mlx_c() turns the
        // `metal` feature into a hard error rather than letting the link fail on a missing
        // framework. A CPU-only MLX is the whole available surface.
        //
        // 9.0 is not derived from a Metal version — there is none to derive one from. It is the
        // oldest watchOS whose SDK carries the C++17 library features MLX's CPU backend uses
        // unconditionally (`std::filesystem` is annotated watchOS 6+, aligned `operator new`
        // watchOS 4+), with margin, and a real build confirms it: aarch64-apple-watchos-sim
        // compiles libmlx and libmlxc at watchOS 9.0, CPU-only.
        //
        // Only that simulator works. No watchOS *device* target builds MLX — on either ABI, and
        // for two unrelated upstream reasons — see the check in build_and_link_mlx_c(). Both
        // arms are kept anyway so the failure is a one-line diagnosis instead of a wall of C++
        // template errors, and so the SDK/deployment-target facts live in one place.
        ("watchos", false) => Some(ApplePlatform {
            deployment_env: "WATCHOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("watchOS"),
            sdk: "watchos",
            metal_version_min_prefix: None,
            min_version: (9, 0),
        }),
        ("watchos", true) => Some(ApplePlatform {
            deployment_env: "WATCHOS_DEPLOYMENT_TARGET",
            cmake_system_name: Some("watchOS"),
            sdk: "watchsimulator",
            metal_version_min_prefix: None,
            min_version: (9, 0),
        }),
        (other, _) => {
            assert!(
                env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple"),
                "mlx-sys: no build configuration for Apple target_os `{other}`. Every Apple \
                 platform needs an explicit SDK, deployment-target variable and Metal \
                 version-min flag; without one, MLX silently configures MLX_METAL_VERSION=0 and \
                 drops every version-gated kernel. Add an arm to apple_platform() (visionOS is \
                 the known gap) or build for a supported target.",
            );
            None
        }
    }
}

/// Apple's name for the target architecture, for `CMAKE_OSX_ARCHITECTURES`.
///
/// Derived from the target *triple* rather than `CARGO_CFG_TARGET_ARCH`, because the watchOS
/// targets `arm64_32-apple-watchos` and `armv7k-apple-watchos` report `aarch64` and `arm` there
/// respectively — both of which would select the wrong slice. Triple arch names are already
/// Apple's own spellings, so only `aarch64` needs translating.
fn apple_arch() -> Option<String> {
    let triple = env::var("TARGET").ok()?;
    let arch = triple.split('-').next()?;
    Some(match arch {
        "aarch64" => "arm64".to_string(),
        other => other.to_string(),
    })
}

/// Resolve `sdk` to an SDK path via `xcrun`, for an explicit `CMAKE_OSX_SYSROOT`.
///
/// Without this, CMake picks the sysroot itself from `CMAKE_SYSTEM_NAME` alone — and
/// `CMAKE_SYSTEM_NAME=iOS` selects the **device** SDK, so a simulator build compiles its C++
/// against iPhoneOS headers while cc-rs injects a simulator `-isysroot` into CXXFLAGS. The two
/// halves then disagree about which platform they are building for.
fn apple_sysroot(sdk: &str) -> Option<String> {
    let output = Command::new("xcrun")
        .args(["--sdk", sdk, "--show-sdk-path"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!path.is_empty()).then_some(path)
}

/// Whole-word search for a linker symbol in concatenated `.tbd` stub text.
///
/// Substring matching would let `_MTLIOErrorDomain` be "found" inside `_MTLIOErrorDomainKey`.
fn tbd_exports(tbd: &str, symbol: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    tbd.match_indices(symbol).any(|(start, _)| {
        let before = tbd[..start].chars().next_back();
        let after = tbd[start + symbol.len()..].chars().next();
        !before.is_some_and(ident) && !after.is_some_and(ident)
    })
}

/// Define, as null, the metal-cpp constants the target SDK does not export.
///
/// `mlx/backend/metal/device.cpp` is metal-cpp's single `MTL_PRIVATE_IMPLEMENTATION` translation
/// unit, so `libmlx.a` carries one reference per metal-cpp private constant — 48 of them at MLX
/// 0.32. Built against a 26.x SDK, metal-cpp resolves those at **link** time:
///
///     _MTL_EXTERN type const MTLsymbol __attribute__((weak_import));
///     type const MTL::symbol = (nullptr != &MTLsymbol) ? MTLsymbol : nullptr;
///
/// Only on an older SDK does it take its `dlsym(RTLD_DEFAULT, ...)` fallback. That choice is
/// gated on SDK *version* (`__MAC_26_0` / `__IPHONE_26_0` / `__TVOS_26_0`) and not on the
/// platform variant — but the iOS and tvOS **simulator** Metal frameworks export strictly fewer
/// symbols than their device counterparts. `weak_import` permits a symbol to disappear at
/// runtime, NOT at link time, so every simulator binary linking libmlx.a dies with
///
///     Undefined symbols: _MTLIOErrorDomain, _MTLTensorDomain
///
/// which is exactly the failure that made this PR turn `MLX_C_BUILD_EXAMPLES` off — the examples
/// were the only final link in the build, so disabling them moved the error into the consuming
/// app instead of fixing it. Correcting `CMAKE_OSX_SYSROOT` does not help: the objects are right,
/// the simulator SDK genuinely has no such symbols.
///
/// A weak null definition reproduces exactly what metal-cpp's own dlsym path yields on these
/// platforms (`MTL::IOErrorDomain == nullptr`), and MLX references neither constant. `weak` so a
/// future SDK that does export them wins the tie instead of colliding.
///
/// The set is derived from the archive and the SDK rather than hardcoded, so a new metal-cpp
/// constant, MLX version or Xcode release cannot silently reintroduce the link failure.
fn shim_unexported_metal_cpp_symbols(platform: &ApplePlatform, lib_dir: &std::path::Path) {
    let archive = lib_dir.join("libmlx.a");
    // The frameworks metal-cpp's private constants live in, which are also the only ones MLX
    // links. A `.tbd` is the linker's view of a framework, so this is the same set of symbols
    // `ld` will be able to resolve.
    let sdk_root = apple_sysroot(platform.sdk).unwrap_or_default();
    let exports: String = ["Metal", "Foundation", "QuartzCore"]
        .iter()
        .filter_map(|fw| {
            std::fs::read_to_string(format!(
                "{sdk_root}/System/Library/Frameworks/{fw}.framework/{fw}.tbd"
            ))
            .ok()
        })
        .collect();
    let nm = Command::new("nm").arg("-m").arg(&archive).output();

    // Any failure here means "we cannot tell which constants this SDK is missing". Carrying on
    // regardless just defers an undefined-symbol error to the consumer's link, naming a symbol
    // nothing in MLX uses — so stop where the cause is still visible.
    assert!(
        !exports.is_empty() && nm.as_ref().is_ok_and(|out| out.status.success()),
        "mlx-sys: could not read the {} SDK stubs under {sdk_root:?} or run `nm` on {}, so the \
         metal-cpp constants this SDK does not export cannot be determined.",
        platform.sdk,
        archive.display(),
    );
    let output = nm.expect("checked by the assert above");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut missing: Vec<&str> = stdout
        .lines()
        .filter(|line| line.contains("(undefined) weak external"))
        .filter_map(|line| line.split_whitespace().next_back())
        .filter(|sym| ["_MTL", "_NS", "_CA"].iter().any(|p| sym.starts_with(p)))
        .filter(|sym| !tbd_exports(&exports, sym))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    if missing.is_empty() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shim = out_dir.join("metal_cpp_absent_symbols.c");
    let mut src = String::from(
        "/* Generated by mlx-sys/build.rs — see shim_unexported_metal_cpp_symbols(). */\n",
    );
    for sym in &missing {
        // Drop the Mach-O leading underscore; the C identifier is what the compiler mangles.
        let name = sym.strip_prefix('_').unwrap_or(sym);
        src.push_str(&format!("__attribute__((weak)) const void *{name} = 0;\n"));
    }
    std::fs::write(&shim, src).expect("Failed to write metal-cpp symbol shim");

    println!(
        "cargo:warning=Defining as null {} metal-cpp constant(s) that the {} SDK does not export: \
         {}. MLX never reads them; without this, linking any binary against MLX for this target \
         fails with an undefined-symbol error.",
        missing.len(),
        platform.sdk,
        missing.join(", "),
    );

    // Emitted after the `mlx`/`mlxc` link directives so the definitions follow the references on
    // the link line.
    cc::Build::new()
        .file(&shim)
        .warnings(false)
        .compile("mlx_metal_cpp_absent");
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

    // sc-18316: retained Rust compile handles may outlive MLX's C++ thread-local compiler cache
    // when both are first initialized in the opposite order. Expose a non-mutating cache touch so
    // downstream Rust TLS owners can establish safe reverse-destruction order before creating
    // their own slot. Keep the pinned upstream mlx-c submodule pristine; this is a pmetal-specific
    // lifecycle bridge until mlx-c exposes an equivalent API.
    let cache_init_patch = std::fs::canonicalize("patches/compile-cache-initialize.patch")
        .expect("find compile-cache-initialize.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&cache_init_patch)
        .status()
        .expect("Failed to run `patch` for compile-cache-initialize.patch");
    assert!(
        status.success(),
        "compile-cache-initialize.patch failed to apply to staged mlx-c (sc-18316)"
    );
    println!("cargo:rerun-if-changed=patches/compile-cache-initialize.patch");

    // sc-18318: expose the pmetal-only exact Q4/Q8 qmm+dense-bias operation
    // added by exact-qmm-bias.patch. Bindgen continues to read the pristine
    // pinned submodule; mlx-sys declares the extension manually.
    let qmm_bias_c_patch = std::fs::canonicalize("patches/exact-qmm-bias-c.patch")
        .expect("find exact-qmm-bias-c.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&qmm_bias_c_patch)
        .status()
        .expect("Failed to run `patch` for exact-qmm-bias-c.patch");
    assert!(
        status.success(),
        "exact-qmm-bias-c.patch failed to apply to staged mlx-c (sc-18318)"
    );
    println!("cargo:rerun-if-changed=patches/exact-qmm-bias-c.patch");

    // sc-18318: expose the exact biased Conv2d/Conv3d implicit-GEMM bridge
    // added by exact-conv-bias.patch.
    let conv_bias_c_patch = std::fs::canonicalize("patches/exact-conv-bias-c.patch")
        .expect("find exact-conv-bias-c.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&conv_bias_c_patch)
        .status()
        .expect("Failed to run `patch` for exact-conv-bias-c.patch");
    assert!(
        status.success(),
        "exact-conv-bias-c.patch failed to apply to staged mlx-c (sc-18318)"
    );
    println!("cargo:rerun-if-changed=patches/exact-conv-bias-c.patch");

    // sc-18318: expose the group-aware exact normalization+affine bridge
    // added by exact-group-norm-affine.patch.
    let group_norm_affine_c_patch =
        std::fs::canonicalize("patches/exact-group-norm-affine-c.patch")
            .expect("find exact-group-norm-affine-c.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&group_norm_affine_c_patch)
        .status()
        .expect("Failed to run `patch` for exact-group-norm-affine-c.patch");
    assert!(
        status.success(),
        "exact-group-norm-affine-c.patch failed to apply to staged mlx-c (sc-18318)"
    );
    println!("cargo:rerun-if-changed=patches/exact-group-norm-affine-c.patch");

    // sc-18318: expose the exact eager SiLU and tanh-GELU bridges added by
    // exact-eager-activations.patch.
    let exact_activations_c_patch =
        std::fs::canonicalize("patches/exact-eager-activations-c.patch")
            .expect("find exact-eager-activations-c.patch");
    let status = Command::new("patch")
        .arg("-p1")
        .arg("-d")
        .arg(&staged)
        .arg("-i")
        .arg(&exact_activations_c_patch)
        .status()
        .expect("Failed to run `patch` for exact-eager-activations-c.patch");
    assert!(
        status.success(),
        "exact-eager-activations-c.patch failed to apply to staged mlx-c (sc-18318)"
    );
    println!("cargo:rerun-if-changed=patches/exact-eager-activations-c.patch");

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
    //   - apple-metal-sdk.patch               : cross-compile the Metal kernels for
    //     non-macOS Apple platforms. MLX's kernel rules hardcode
    //     `xcrun -sdk macosx metal` and `-mmacosx-version-min`, so a
    //     CMAKE_SYSTEM_NAME=iOS build still emitted a **macOS** metallib: the C++
    //     cross-compiled, the binary linked, and the artifact would have failed at
    //     `newLibraryWithURL` on device. (MLX's root CMakeLists already guards its
    //     SDK probe on `CMAKE_SYSTEM_NAME STREQUAL "Darwin"` and falls back to
    //     MLX_METAL_VERSION 0, so upstream anticipated non-macOS configuration —
    //     the kernel rules just never got the matching branch.)
    //     The patch takes the SDK and version-min flag from MLX_METAL_SDK /
    //     MLX_METAL_VERSION_MIN_PREFIX (set by apple_platform() above) in both the
    //     per-kernel compile and the metallib link, and adds an Apple-cross arm to
    //     the root probe so MLX_METAL_VERSION is detected rather than pinned to 0.
    //     Those two variables are deliberately NOT derived inside CMake: device and
    //     simulator are different SDKs taking different flags
    //     (iphoneos/-mios-version-min= vs iphonesimulator/-mios-simulator-version-min=)
    //     and CMAKE_SYSTEM_NAME is `iOS` for both, so CMake cannot tell them apart.
    //     A simulator build driven off CMAKE_SYSTEM_NAME alone produces
    //     air64-apple-ios kernels that will not load in the simulator.
    //     Both defaults stay `macosx`/`-mmacosx-version-min=`, so a plain macOS
    //     build is byte-identical to upstream; a non-macOS Apple build that reaches
    //     the kernel rules without them set is a FATAL_ERROR rather than a silent
    //     MLX_METAL_VERSION 0 (which would drop every version-gated kernel).
    //     Deployment target matters here: iOS/tvOS 16 -> Metal 300, 17 -> 310,
    //     18 -> 320. MLX's macOS floor (14.0) is Metal 310, so 16 is BELOW the
    //     baseline its kernels assume; apple_platform() floors both at 18.0 (Metal
    //     320) and the patch asserts >= 310 after probing.
    //     That also keeps `fence` coherent — the kernel is built only at
    //     MLX_METAL_VERSION >= 320 while fence.cpp's runtime guard is
    //     `__builtin_available(macOS 15, iOS 18, *)`, so a lower floor could satisfy
    //     the runtime check with the kernel absent.
    //     The NAX gate fails safe off macOS: it needs Metal >= 400 AND
    //     MACOS_SDK_VERSION >= 26.2, and the latter is unset there.
    //     watchOS is not covered: its SDKs ship no Metal framework at all, so
    //     build.rs rejects the `metal` feature there before cmake runs.
    //   - apple-cpu-no-jit.patch              : route tvOS/watchOS/visionOS to the
    //     same no-op CPU compiler upstream already uses for iOS.
    //     mlx/backend/cpu/jit_compiler.cpp shells out to a host compiler via
    //     `std::system()`, which libc marks __IOS_PROHIBITED __TVOS_PROHIBITED
    //     __WATCHOS_PROHIBITED — there is no process spawning in these sandboxes.
    //     Upstream guards it with `if(IOS)`, but CMake defines only IOS and leaves
    //     TVOS/WATCHOS unset even when CMAKE_SYSTEM_NAME is tvOS/watchOS, so those
    //     platforms still compiled the JIT path and failed with
    //     "'system' is unavailable: not available on tvOS". The patch matches on
    //     CMAKE_SYSTEM_NAME instead. macOS and iOS take exactly the branch they did
    //     before.
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
    // (basename, required, applied_probe). `required = true` means the build MUST fail if the
    // patch does not apply; `false` means best-effort (a fully-failing patch is a safe no-op).
    // `applied_probe` is reserved for sequential extension patches that overlap files: after a
    // later patch lands, an earlier patch may no longer reverse-apply even though its complete,
    // unique surface is present. The probe checks multiple unique artifacts and only skips when
    // that complete surface is already installed.
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
        ("patches/metallib-search-path.patch", true, None),
        ("patches/command-buffer-recoverable.patch", true, None),
        ("patches/pad-copy-int64.patch", true, None),
        ("patches/thread-shared-streams.patch", true, None),
        ("patches/thread-safe-eval.patch", true, None),
        ("patches/apple-metal-sdk.patch", true, None),
        ("patches/apple-cpu-no-jit.patch", true, None),
        (
            "patches/exact-qmm-bias.patch",
            true,
            Some(
                "grep -Fq 'quantized_matmul_bias(' mlx/ops.h && grep -Fq 'affine_qmm_bias_t' mlx/backend/metal/kernels/quantized.metal && grep -Fq 'affine_qmm_bias_t_nax' mlx/backend/metal/kernels/quantized_nax.metal",
            ),
        ),
        (
            "patches/exact-conv-bias.patch",
            true,
            Some(
                "grep -Fq 'conv_general_bias(' mlx/ops.h && grep -Fq 'HAS_OUTPUT_BIAS' mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h && grep -Fq 'HAS_OUTPUT_BIAS' mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.h",
            ),
        ),
        ("patches/exact-conv-bias-nojit.patch", true, None),
        (
            "patches/exact-group-norm-affine.patch",
            true,
            Some(
                "test -f mlx/backend/metal/kernels/group_norm.metal && grep -Fq 'group_norm_affine(' mlx/fast.h && grep -Fq 'GroupNormAffine::eval_gpu' mlx/backend/metal/normalization.cpp",
            ),
        ),
        (
            "patches/exact-eager-activations.patch",
            true,
            Some(
                "test -f mlx/backend/metal/kernels/exact_activations.metal && grep -Fq 'silu_exact(' mlx/fast.h && grep -Fq 'ExactActivation::eval_gpu' mlx/backend/metal/normalization.cpp",
            ),
        ),
        ("patches/exact-backend-stubs.patch", true, None),
        (
            "patches/exact-epilogue-dispatcher-tests.patch",
            true,
            Some(
                "grep -Fq 'constexpr int ODD_M = 257;' tests/ops_tests.cpp && grep -Fq 'bool noncontiguous = false' tests/ops_tests.cpp",
            ),
        ),
        (
            "patches/exact-dispatch-and-affine-lifetime.patch",
            true,
            Some(
                "grep -Fq 'const bool eval_row_contiguous' mlx/ops.cpp && grep -Fq 'if (weight_copied)' mlx/backend/metal/normalization.cpp && grep -Fq 'constexpr int BATCH_M = 33;' tests/ops_tests.cpp",
            ),
        ),
        (
            "patches/exact-qmm-empty-dimensions.patch",
            true,
            Some(
                "grep -Fq 'w_inner_dims <= 0' mlx/ops.cpp && awk '/if \\(M < 32\\)/ { m=NR } /int B = x_cast.size\\(\\) \\/ M \\/ w_inner_dims/ { b=NR } END { exit !(m && b && m < b) }' mlx/ops.cpp && grep -Fq 'zeros({0, K})' tests/ops_tests.cpp && grep -Fq 'zeros({32, 0})' tests/ops_tests.cpp",
            ),
        ),
        (
            "patches/exact-supported-dispatch.patch",
            true,
            Some(
                "grep -Fq 'Input and weight must promote to float32' mlx/ops.cpp && grep -Fq 'Only float32, float16, and bfloat16 are' mlx/ops.cpp && grep -Fq 'zeros({1, 8, 8, 16}, float64)' tests/ops_tests.cpp",
            ),
        ),
        (
            "patches/exact-strict-metal-output.patch",
            true,
            Some(
                "test \"$(grep -Fc 's.device != Device::gpu || !metal::is_available()' mlx/fast.cpp)\" -eq 2 && grep -Fq 'Empty weight or output dimensions' mlx/ops.cpp && grep -Fq 'test exact fast primitives reject without Metal' tests/ops_tests.cpp && grep -Fq 'zeros({2, 33, K})' tests/ops_tests.cpp",
            ),
        ),
    ];
    // sc-12780 idempotency guard: CMake FetchContent may re-run PATCH_COMMAND against an
    // mlx-src that is ALREADY patched (e.g. an incremental rebuild that does not re-fetch).
    // `git apply` is not idempotent — re-applying an applied patch fails "patch does not apply".
    // So each patch is guarded: independent patches use a reverse-apply check; sequential exact
    // epilogue patches that overlap files use complete-surface probes because a later patch can
    // invalidate an earlier reverse check. An already-present patch is skipped; otherwise it is
    // applied forward. A `required` patch that genuinely cannot apply aborts the build. This keeps
    // re-runs a no-op while still hard-failing on a real apply failure — the point of required=true.
    let mut script = String::from(
        "#!/bin/sh\n\
         # Generated by mlx-sys/build.rs. Apply each MLX source patch individually,\n\
         # idempotently (sc-12780). cwd is the fetched mlx-src (a git repo); patches\n\
         # live next to this script.\n\
         d=\"$(dirname \"$0\")\"\n",
    );
    for (pf, required, applied_probe) in patch_files {
        let name = std::path::Path::new(pf)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        std::fs::copy(pf, patches_dir.join(&name))
            .unwrap_or_else(|e| panic!("Failed to copy {pf}: {e}"));
        if required {
            let probe = applied_probe
                .map(|probe| format!("{probe}; then\n  "))
                .unwrap_or_else(|| {
                    format!("git apply --reverse --check \"$d/{name}\" 2>/dev/null; then\n  ")
                });
            script.push_str(&format!(
                "if {probe}\
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
    for (pf, _required, _applied_probe) in patch_files {
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
    let deployment_target = platform.as_ref().map(|platform| {
        let target = resolve_deployment_target(platform);
        // `env::set_var` reaches the cmake/cc child processes, but NOT rustc's link of a
        // downstream binary — that is a separate cargo invocation with its own environment. Off
        // macOS an unset deployment target there drops __chkstk_darwin (libSystem, iOS 12+) and
        // surfaces as a bare undefined-symbol error in the consuming crate, with nothing
        // pointing back to here. Say so while the cause is still identifiable.
        if platform.cmake_system_name.is_some() && env::var(platform.deployment_env).is_err() {
            println!(
                "cargo:warning={env} is not set. MLX is being built at {target}, but cargo will \
                 not pass that to the final link — set {env}={target} in your environment or \
                 .cargo/config.toml, or expect an undefined `___chkstk_darwin` when linking.",
                env = platform.deployment_env,
            );
        }
        env::set_var(platform.deployment_env, &target);
        target
    });

    // watchOS has no Metal framework in either its device or simulator SDK, so MLX_BUILD_METAL
    // cannot work there at all. Fail here with the reason rather than at link time with a
    // missing-framework error.
    #[cfg(feature = "metal")]
    if let Some(platform) = &platform {
        assert!(
            platform.metal_version_min_prefix.is_some(),
            "mlx-sys: the `metal` feature is enabled, but the {} SDK ships no Metal framework. \
             Build with `--no-default-features` (plus any other features you need) for a \
             CPU-only MLX on this target.",
            platform.sdk,
        );
    }

    // No watchOS *device* target builds MLX, and the two ABIs fail for unrelated reasons — so
    // neither "it's the 32-bit ABI" nor "it's the SIMD layer" is the whole story. Measured
    // against MLX 0.32 / Xcode 26.5, CPU-only, with and without the `accelerate` feature:
    //
    //   arm64_32, armv7k (ILP32)     mlx/backend/cpu/simd/ assumes LP64 — `simd::Vector<long,
    //                                N>` has no `packed_t` and `simd::Simd<long long, N>` has
    //                                no valid construction. ~30 errors in binary.h, reduce.cpp
    //                                and accelerate_simd.h.
    //   aarch64-apple-watchos (LP64) compiles past that and dies on MLX's half-precision types:
    //                                the watchOS SDK makes `__bf16` and `__fp16` distinct
    //                                native types, so compiled.h has an ambiguous `operator<<`
    //                                and libc++ `copy()` rejects the assignment.
    //
    // Both are upstream MLX ports. The watchOS *simulator* is unaffected and does build, which
    // is why the watchOS arms above exist at all.
    if let Some(platform) = &platform {
        assert!(
            platform.sdk != "watchos",
            "mlx-sys: MLX does not build for watchOS devices on any ABI — the ILP32 targets \
             (arm64_32, armv7k) fail in MLX's LP64-only CPU SIMD layer, and the LP64 target \
             (aarch64-apple-watchos) fails on its `__bf16`/`__fp16` handling. Both need fixing \
             upstream in MLX. `aarch64-apple-watchos-sim` does build, CPU-only.",
        );
    }

    // Separately from watchOS: MLX's CPU SIMD layer is LP64-only, so no 32-bit Apple target can
    // work. Beyond the watchOS ABIs above this only catches targets Apple retired years ago
    // (i386/armv7s-apple-ios, i686-apple-darwin), but it costs nothing and keeps the failure a
    // one-line diagnosis instead of a ten-minute build ending in C++ template errors.
    if platform.is_some() {
        assert!(
            env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() != Ok("32"),
            "mlx-sys: MLX does not build for 32-bit-pointer Apple targets — its CPU SIMD layer \
             (mlx/backend/cpu/simd/) assumes LP64 and does not compile under ILP32.",
        );
    }

    let mlx_c_src = prepare_mlx_c_source();
    let mut config = Config::new(&mlx_c_src);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // mlx-c's example .app targets are never linked into this crate, so building them is pure
    // cost on every platform.
    //
    // They were originally turned off because they failed on the iOS simulator with undefined
    // `_MTLIOErrorDomain` — but that was the examples correctly reporting a real defect in what
    // this crate ships, not a problem with the examples: they were the only final link in the
    // build, and every consumer binary hit the same error.
    // shim_unexported_metal_cpp_symbols() fixes the cause, so this is now only about build time.
    // The cross-apple CI workflow links a real binary per target to keep that honest.
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");

    if let Some(platform) = &platform {
        // CMAKE_OSX_DEPLOYMENT_TARGET is the knob for every Apple platform, not just macOS.
        config.define(
            "CMAKE_OSX_DEPLOYMENT_TARGET",
            deployment_target.as_deref().unwrap_or_default(),
        );

        // Hand MLX the SDK and version-min flag for THIS platform. apple-metal-sdk.patch reads
        // both: the per-kernel compile, the metallib link and the root MLX_METAL_VERSION probe
        // all default to `-sdk macosx` / `-mmacosx-version-min` upstream, which cross-compiles
        // to a macOS metallib that loads nowhere else. The patch hard-errors on a non-macOS
        // Apple system with these unset rather than falling back to MLX_METAL_VERSION 0.
        if let Some(prefix) = platform.metal_version_min_prefix {
            config.define("MLX_METAL_SDK", platform.sdk);
            config.define("MLX_METAL_VERSION_MIN_PREFIX", prefix);
        }

        if let Some(system_name) = platform.cmake_system_name {
            // Cross-compiling. cmake-rs sets CMAKE_SYSTEM_NAME/CMAKE_SYSTEM_PROCESSOR itself,
            // but only when neither we nor the environment already defined the former — and it
            // sets *both or neither*. Since we need the platform decision here anyway (for the
            // sysroot and the Metal SDK), set both explicitly rather than leaving
            // CMAKE_SYSTEM_PROCESSOR unset as a side effect of naming the system.
            config.define("CMAKE_SYSTEM_NAME", system_name);
            if let Some(arch) = apple_arch() {
                config.define("CMAKE_SYSTEM_PROCESSOR", &arch);
                config.define("CMAKE_OSX_ARCHITECTURES", &arch);
            }
            // CMAKE_SYSTEM_NAME alone does not distinguish device from simulator — CMake maps
            // `iOS` to the iPhoneOS SDK either way — so name the sysroot explicitly.
            if let Some(sysroot) = apple_sysroot(platform.sdk) {
                config.define("CMAKE_OSX_SYSROOT", &sysroot);
            } else {
                // Fall back to the SDK name, which CMake also accepts, so a failing xcrun does
                // not silently leave the device SDK selected.
                config.define("CMAKE_OSX_SYSROOT", platform.sdk);
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

        // Must come after the Metal/mlx link directives — see the function's doc comment.
        if let Some(platform) = &platform {
            if platform.cmake_system_name.is_some() {
                shim_unexported_metal_cpp_symbols(platform, &dst.join("build/lib"));
            }
        }
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
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
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
