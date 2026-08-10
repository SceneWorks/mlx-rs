#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Added to the staged mlx-c source by compile-cache-initialize.patch. Bindgen reads the pristine
// pinned submodule, so declare these pmetal lifecycle extensions explicitly.
extern "C" {
    pub fn mlx_detail_compile_initialize_cache() -> ::std::os::raw::c_int;
    pub fn mlx_detail_compile_acquire_cache_token() -> *mut ::std::os::raw::c_void;
    pub fn mlx_detail_compile_erase_with_cache_token(
        token: *mut ::std::os::raw::c_void,
        fun_id: usize,
    ) -> ::std::os::raw::c_int;
}
