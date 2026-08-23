# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

## Prebuilt MLX (`PMETAL_MLX_PREBUILT_DIR`)

`build.rs` normally stages `src/mlx-c`, applies the patches, fetches MLX and runs cmake
(≈6 minutes on a 3-core hosted Mac, plus a network fetch). Every such build writes a
`pmetal-mlx-prebuilt.txt` manifest next to the archives in `build/lib`, and the
`prebuilt-mlx` workflow publishes that directory for each (deployment target × Debug/Release)
cell as `pmetal-mlx-<sha12>-<target>-dt<dt>-<Debug|Release>-<features>.tar.zst` on the
GitHub Release `prebuilt-<sha12>` of the commit it built.

To use one: extract it and point the variable at the directory:

```sh
export PMETAL_MLX_PREBUILT_DIR=/path/to/extracted
cargo build   # mlx-sys links libmlx.a / libmlxc.a / mlx.metallib from there; cmake is skipped
```

The manifest is a key, not a hint. `build.rs` compares `fingerprint` (SHA-256 over `build.rs`,
`Cargo.toml`, `patches/` and `src/mlx-c`), `target`, `deployment_target`, `features` and
`build_type` against its own values and **fails the build** on any difference — a libmlx built
for the wrong deployment target links fine and miscompiles the NAX kernels at runtime, so there
is deliberately no silent fallback. Unset the variable to build from source. Bindgen always runs
against the headers in `src/mlx-c`, so the tarball carries no headers.
