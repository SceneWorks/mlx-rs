// Standard-library-only driver of the SAME generator build.rs imports.
#[path = "../patch_series.rs"]
mod patch_series;
use std::{fs, path::PathBuf};
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    assert_eq!(args.len(), 4, "sys-root output-patch-dir hex-key");
    let root = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let build = fs::read_to_string(root.join("build.rs")).unwrap();
    let table = build
        .split("let patch_files = [")
        .nth(1)
        .unwrap()
        .split("\n    ];")
        .next()
        .unwrap();
    fs::create_dir_all(&out).unwrap();
    let mut patches = Vec::new();
    for line in table.lines() {
        let line = line.trim();
        if let Some(path) = line.strip_prefix('"').and_then(|s| s.strip_suffix("\",")) {
            let name = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            let text = fs::read_to_string(root.join(path)).unwrap();
            fs::write(out.join(&name), &text).unwrap();
            patches.push((name, text));
        }
    }
    let baseline = fs::read_to_string(root.join("patches/native-upstream-baseline.txt")).unwrap();
    let patched = fs::read_to_string(root.join("patches/native-patched-baseline.txt")).unwrap();
    fs::write(
        out.join("apply_patches.sh"),
        patch_series::script(&patches, &baseline, &patched),
    )
    .unwrap();
    fs::write(
        out.join("cmake-declaration.txt"),
        patch_series::cmake_declaration(args[3].to_str().unwrap()),
    )
    .unwrap();
}
