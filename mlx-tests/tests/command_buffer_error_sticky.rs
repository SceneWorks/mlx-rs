//! A command buffer that fails in the MIDDLE of an eval must fail that eval.
//!
//! Upstream MLX 0.32.0 records a failed buffer's error into the per-encoder `error_` from Metal's
//! completion-handler thread, but only AFTER calling the scheduler's completion callback — which
//! wakes a host thread throttled on in-flight buffers — and `get_command_encoder()` resets
//! `error_` whenever the host opens its next encoder. When the failed buffer's handler ran before
//! that reset, the error was erased; the buffer carrying the eval's signal event then completed
//! cleanly and `eval` returned corrupted arrays as `Ok`. Seen in production as a Qwen-Image render
//! that ran through 14 GPU-watchdog timeouts and returned a coherent but wrong image.
//!
//! Two more upstream paths dropped the same error: the GPU signals a successful buffer's events
//! BEFORE its completion handler poisons them, so a host wait could check for an error too early;
//! and `array::wait()` skipped the check entirely when the event was already signaled
//! (`is_available()` detaches it). `mlx-sys/patches/sticky-command-buffer-error.patch` keeps the
//! error until a host wait reports it, makes a host wait also wait for the signaling buffer's
//! handler, and always checks the event in `array::wait()`.
//!
//! This test drives the erased-error ordering deterministically through the patch's debug-only hook
//! (`mlx_pmetal_test_fail_next_intermediate_command_buffer`, `!NDEBUG` builds): the next buffer
//! committed WITHOUT a signal event is failed, and the committing host thread blocks until that
//! buffer's handler has finished — so the error is recorded before the eval opens its next encoder,
//! exactly the ordering that erased it upstream.
//!
//! This file holds a single test on purpose: `MLX_MAX_OPS_PER_BUFFER` is read once, when the Metal
//! device is created, and must be set before any MLX call in the process.

use mlx_rs::{array, Array};

extern "C" {
    fn mlx_pmetal_test_fail_next_intermediate_command_buffer();
}

/// A lazy chain of `n` dependent adds — `n` separate primitives, so with a small
/// `MLX_MAX_OPS_PER_BUFFER` one eval spans many command buffers.
fn chain(n: usize) -> Array {
    let mut x = array!([1.0f32, 2.0, 3.0]);
    for _ in 0..n {
        x = x.add(&array!([1.0f32, 1.0, 1.0])).expect("add");
    }
    x
}

#[test]
fn intermediate_command_buffer_failure_fails_the_eval() {
    // SAFETY: set before the first MLX call; this binary runs no other test.
    unsafe { std::env::set_var("MLX_MAX_OPS_PER_BUFFER", "2") };

    // 1. Control: the same multi-buffer eval succeeds and computes the right values.
    let clean = chain(64);
    mlx_rs::transforms::eval([&clean]).expect("a clean multi-buffer eval should succeed");
    assert_eq!(clean.as_slice::<f32>(), &[65.0, 66.0, 67.0]);

    // 2. Fail one buffer in the middle of the next eval; every later buffer succeeds.
    unsafe { mlx_pmetal_test_fail_next_intermediate_command_buffer() };
    let poisoned = chain(64);
    let err = mlx_rs::transforms::eval([&poisoned])
        .expect_err("a failed intermediate command buffer must fail its eval, not return Ok");
    assert!(
        err.what().contains("synthetic intermediate-buffer failure"),
        "unexpected error surfaced: {}",
        err.what()
    );

    // 3. Reported once: the stream recovers, so later, unrelated work is not poisoned.
    let after = chain(64);
    mlx_rs::transforms::eval([&after]).expect("eval should recover once the error is reported");
    assert_eq!(after.as_slice::<f32>(), &[65.0, 66.0, 67.0]);
}
