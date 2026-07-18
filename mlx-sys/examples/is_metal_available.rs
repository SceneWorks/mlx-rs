// sc-12940: refers to the lib by its upstream name `mlx_sys` — kept working by
// `[lib] name = "mlx_sys"` in this crate's Cargo.toml. Do not rename these
// paths to `pmetal_mlx_sys` (PR #17 briefly did on its own branch; with the
// [lib] rename in place that name no longer resolves).
fn main() {
    let mut is_available = false;
    let status = unsafe { mlx_sys::mlx_metal_is_available(&mut is_available as *mut bool) };
    assert_eq!(status, 0);
    println!("{is_available:?}");
}
