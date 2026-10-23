//! Additive Rust/C/native owner-retirement ABI smoke (CPU only, no GPU submission).
//! Real retained-owner/destructor and Metal tests are in the patched native test suite.
use mlx_rs::{array, Device, Stream};

#[test]
fn cpu_stream_retirement_is_fallible_repeatable_and_preserves_values() {
    Device::set_default(&Device::cpu());
    let stream = Stream::cpu();
    let value = array!([1.0f32, 2.0, 3.0]);
    let result = value.add(&array!([4.0f32, 5.0, 6.0])).expect("CPU add");
    mlx_rs::transforms::eval([&result]).expect("CPU eval");
    let fence = stream
        .retirement_fence()
        .expect("capture completed CPU range");
    fence.wait().expect("observe CPU owner retirement");
    fence.wait().expect("rewait same immutable receipt");
    stream
        .synchronize_and_retire()
        .expect("new retirement range");
    assert_eq!(result.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
}
