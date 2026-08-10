#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Added to the staged mlx-c source by compile-cache-initialize.patch. Bindgen reads the pristine
// pinned submodule, so declare this pmetal lifecycle extension explicitly.
extern "C" {
    pub fn mlx_detail_compile_initialize_cache() -> ::std::os::raw::c_int;
}
