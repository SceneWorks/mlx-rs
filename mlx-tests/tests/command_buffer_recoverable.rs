//! sc-5009: a Metal command-buffer error (e.g. the GPU watchdog
//! `kIOGPUCommandBufferCallbackErrorTimeout`, or an out-of-memory status) must
//! surface from `eval` as a recoverable `mlx_rs::Exception` (`Result::Err`)
//! instead of escaping a Metal completion-handler thread and calling
//! `std::terminate` (SIGABRT).
//!
//! In MLX core such an error is detected inside a `MTL::CommandBuffer`
//! completion handler that runs on an internal Metal thread. The pmetal patch
//! (`mlx-sys/patches/command-buffer-recoverable.patch`) makes that handler
//! *record* the error instead of throwing, and re-throws it synchronously on
//! the waiting (calling) thread (`Event::wait` / `CommandEncoder::synchronize`).
//!
//! Actually tripping the real GPU watchdog is non-deterministic and can disrupt
//! the host display, so this test drives the exact same recovery path through a
//! debug-only C hook compiled into MLX core
//! (`mlx_pmetal_test_inject_command_buffer_error`, gated on `!NDEBUG`) that
//! records a synthetic error into the same slot a real fault would. Reaching any
//! assertion below at all proves the process did NOT abort; the `Err` proves the
//! error is recoverable.

use mlx_rs::array;

extern "C" {
    /// Debug-only hook in MLX core (`mlx/backend/metal/eval.cpp`). Records a
    /// synthetic async command-buffer error into the slot the completion
    /// handler writes to. Present only in non-release (`!NDEBUG`) MLX builds.
    fn mlx_pmetal_test_inject_command_buffer_error(msg: *const std::os::raw::c_char);
}

#[test]
fn command_buffer_error_surfaces_as_err_not_terminate() {
    // 1. A clean eval succeeds and does not spuriously surface an error.
    let a = array!([1.0f32, 2.0, 3.0]);
    let b = array!([4.0f32, 5.0, 6.0]);
    let c = a.add(&b).expect("add");
    mlx_rs::transforms::eval([&c]).expect("a clean eval should succeed");

    // 2. Record a synthetic async command-buffer error, exactly as a Metal
    //    completion handler now does (instead of throwing -> terminate).
    let msg = std::ffi::CString::new(
        "[METAL] Command buffer execution failed: kIOGPUCommandBufferCallbackErrorTimeout",
    )
    .unwrap();
    unsafe { mlx_pmetal_test_inject_command_buffer_error(msg.as_ptr()) };

    // 3. The next eval must SURFACE it as a catchable Err on this thread.
    let d = array!([7.0f32, 8.0, 9.0]);
    let e = d.multiply(&array!([2.0f32, 2.0, 2.0])).expect("multiply");
    let err = mlx_rs::transforms::eval([&e])
        .expect_err("a recorded Metal command-buffer error must surface as Err, not terminate");
    assert!(
        err.what().contains("Command buffer execution failed"),
        "unexpected error surfaced: {}",
        err.what()
    );

    // 4. The error is drained, not sticky: a subsequent clean eval recovers,
    //    proving the failure did not permanently wedge the stream.
    let f = array!([1.0f32]).add(&array!([1.0f32])).expect("add");
    mlx_rs::transforms::eval([&f]).expect("eval should recover after the error is taken");
}
