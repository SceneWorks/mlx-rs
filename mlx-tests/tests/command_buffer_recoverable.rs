//! sc-5009 / sc-12786: a Metal command-buffer error (e.g. the GPU watchdog
//! `kIOGPUCommandBufferCallbackErrorTimeout`, or an out-of-memory status) must
//! surface from `eval` as a recoverable `mlx_rs::Exception` (`Result::Err`)
//! instead of escaping a Metal completion-handler thread and calling
//! `std::terminate` (SIGABRT).
//!
//! MLX 0.32.0 provides that recovery UPSTREAM: the completion handler in
//! `CommandEncoder::commit()` (device.cpp) records the error into the
//! per-encoder `error_` and POISONS the buffer's signal events via
//! `EventImpl::set_error()` instead of throwing, and the poisoned error is
//! re-thrown synchronously on the waiting (host) thread by
//! `EventImpl::wait()` -> `check_error()` (or `CommandEncoder::synchronize()`).
//! (Before 0.32.0 the pmetal fork carried its own record-not-throw patch;
//! sc-12780 retired that as redundant once upstream provided the behavior.)
//!
//! Actually tripping the real GPU watchdog is non-deterministic and can disrupt
//! the host display, so this test drives the recovery path through a debug-only
//! C hook compiled into MLX core
//! (`mlx_pmetal_test_inject_command_buffer_error`, gated on `!NDEBUG`, added by
//! `mlx-sys/patches/command-buffer-recoverable.patch`). The hook records a
//! synthetic error that `Event::signal()` drains on the host thread and injects
//! through the REAL upstream poison mechanism — the same
//! `EventImpl::set_error()` call the completion handler's poison loop makes on
//! a real command-buffer error. From there the error surfaces through UNPATCHED
//! upstream code only: `eval` host-waits on the poisoned event and
//! `EventImpl::wait()` -> `check_error()` re-throws it on the calling thread
//! (sc-12786 — previously the hook was drained by a patched-in check in the
//! outer `Event::wait()` BEFORE `EventImpl::wait()`, which proved catchability
//! via a synthetic proxy but would have stayed green even if upstream's
//! `set_error`/`check_error` plumbing regressed).
//!
//! Injection happens at `Event::signal()` (host thread, before the host wait
//! begins) rather than in the completion handler itself because a
//! successfully-completed buffer signals its events before the handler runs on
//! Metal's callback thread — a handler-side injection would race the waiter's
//! `check_error()` and flake. A real command-buffer error has no such race:
//! Metal does not signal the events, and the handler poisons them before
//! manually signaling.
//!
//! Two pieces of the upstream mechanism remain unexercised here, deliberately:
//! the completion handler's record half (`cbuf->status()` -> `error_` -> poison
//! loop) cannot be driven without a real GPU fault, and the
//! `CommandEncoder::synchronize()` re-raise variant is not what `eval`'s
//! host-wait uses. This test guards the poison/re-raise half
//! (`set_error` -> `wait`/`check_error`) that `eval` depends on.
//!
//! Reaching any assertion below at all proves the process did NOT abort; the
//! `Err` proves the poisoned event's error is re-raised catchably on the
//! waiting thread.

use mlx_rs::array;

extern "C" {
    /// Debug-only hook in MLX core (`mlx/backend/metal/eval.cpp`). Records a
    /// synthetic async command-buffer error that `Event::signal()` (event.cpp)
    /// drains and injects via the real `EventImpl::set_error()` — poisoning the
    /// next signaled event exactly as the `CommandEncoder::commit()` completion
    /// handler does for a real command-buffer error. The waiting thread then
    /// re-raises it through unpatched upstream code
    /// (`EventImpl::wait()` -> `check_error()`). Present only in non-release
    /// (`!NDEBUG`) MLX builds.
    fn mlx_pmetal_test_inject_command_buffer_error(msg: *const std::os::raw::c_char);
}

#[test]
fn command_buffer_error_surfaces_as_err_not_terminate() {
    // 1. A clean eval succeeds and does not spuriously surface an error.
    let a = array!([1.0f32, 2.0, 3.0]);
    let b = array!([4.0f32, 5.0, 6.0]);
    let c = a.add(&b).expect("add");
    mlx_rs::transforms::eval([&c]).expect("a clean eval should succeed");

    // 2. Record a synthetic async command-buffer error. The next eval's signal
    //    event gets poisoned via the real `EventImpl::set_error`, exactly as
    //    the commit() completion handler poisons signal events on a real
    //    command-buffer error (instead of throwing -> terminate).
    let msg = std::ffi::CString::new(
        "[METAL] Command buffer execution failed: kIOGPUCommandBufferCallbackErrorTimeout",
    )
    .unwrap();
    unsafe { mlx_pmetal_test_inject_command_buffer_error(msg.as_ptr()) };

    // 3. The next eval must SURFACE it as a catchable Err on this thread, via
    //    upstream's own EventImpl::wait -> check_error re-raise. If upstream
    //    ever regresses the poison/re-raise plumbing (set_error dropped,
    //    check_error no longer called from EventImpl::wait), the poisoned
    //    event is silently ignored, this eval returns Ok, and expect_err
    //    fails the test — that regression guard is the point of sc-12786.
    let d = array!([7.0f32, 8.0, 9.0]);
    let e = d.multiply(&array!([2.0f32, 2.0, 2.0])).expect("multiply");
    let err = mlx_rs::transforms::eval([&e])
        .expect_err("a poisoned signal event must surface as Err, not terminate");
    assert!(
        err.what().contains("Command buffer execution failed"),
        "unexpected error surfaced: {}",
        err.what()
    );

    // 4. The error is drained, not sticky: `check_error()` takes the error
    //    exactly once (atomic exchange) and later evals wait on fresh events,
    //    so a subsequent clean eval recovers — the failure did not permanently
    //    wedge the stream.
    let f = array!([1.0f32]).add(&array!([1.0f32])).expect("add");
    mlx_rs::transforms::eval([&f]).expect("eval should recover after the error is taken");
}
