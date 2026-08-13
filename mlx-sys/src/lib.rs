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
    pub fn mlx_pmetal_quantized_matmul_bias(
        res: *mut mlx_array,
        x: mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        output_bias: mlx_array,
        transpose: bool,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        stream: mlx_stream,
    ) -> ::std::os::raw::c_int;
}
