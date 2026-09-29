//! A GPU command-buffer error seen by a CPU-stream wait must fail the eval, not abort the process.
//!
//! When an op on the CPU stream consumes an array computed on the GPU, MLX enqueues a wait on the
//! GPU's fence event onto the CPU stream's scheduler thread. Upstream's `EventImpl::wait` rethrows
//! the event's command-buffer error, and on the scheduler thread nothing catches it, so the process
//! called `std::terminate` (SIGABRT, rc 134). Seen in production as a GPU-watchdog timeout during
//! weight loading that aborted the worker instead of returning `Err`.
//!
//! `mlx-sys/patches/cpu-stream-wait-error.patch` makes that wait non-throwing: it records the error
//! as the CPU stream's pending error, the stream's later signals carry it to the host, and the host
//! wait that reports it clears it.
//!
//! The error is injected deterministically through the debug-only hook
//! `mlx_pmetal_test_inject_command_buffer_error` (`!NDEBUG` builds): it poisons the next event
//! signaled on the host, which in the eval below is the GPU fence event the CPU op waits on.
//!
//! This file holds a single test on purpose: the injection hook is process-global.

use mlx_rs::{array, Array, StreamOrDevice};

extern "C" {
    fn mlx_pmetal_test_inject_command_buffer_error(msg: *const std::os::raw::c_char);
}

/// A GPU add whose result is consumed by an add on the CPU stream: the CPU stream waits on the
/// GPU's fence event.
fn gpu_then_cpu() -> (Array, Array) {
    let gpu = array!([1.0f32, 2.0, 3.0])
        .add(&array!([1.0f32, 1.0, 1.0]))
        .expect("gpu add");
    let cpu = gpu
        .add_device(&array!([10.0f32, 10.0, 10.0]), StreamOrDevice::cpu())
        .expect("cpu add");
    (gpu, cpu)
}

#[test]
fn gpu_error_seen_by_a_cpu_stream_wait_fails_the_eval() {
    // 1. Control: the same cross-stream eval succeeds and computes the right values.
    let (_, clean) = gpu_then_cpu();
    mlx_rs::transforms::eval([&clean]).expect("a clean GPU->CPU eval should succeed");
    assert_eq!(clean.as_slice::<f32>(), &[12.0, 13.0, 14.0]);

    // 2. Poison the GPU fence event the CPU stream waits on. Before the fix the CPU-stream wait
    //    threw on MLX's scheduler thread and the process aborted before `eval` returned.
    let msg = std::ffi::CString::new("[METAL] Command buffer execution failed: synthetic").unwrap();
    unsafe { mlx_pmetal_test_inject_command_buffer_error(msg.as_ptr()) };
    let (_, poisoned) = gpu_then_cpu();
    let err = mlx_rs::transforms::eval([&poisoned])
        .expect_err("a GPU error seen by a CPU-stream wait must fail the eval");
    assert!(
        err.what().contains("synthetic"),
        "unexpected error surfaced: {}",
        err.what()
    );

    // 3. Reported once: both streams recover, so later, unrelated work is not poisoned.
    let (gpu, cpu) = gpu_then_cpu();
    mlx_rs::transforms::eval([&cpu, &gpu]).expect("eval should recover once the error is reported");
    assert_eq!(gpu.as_slice::<f32>(), &[2.0, 3.0, 4.0]);
    assert_eq!(cpu.as_slice::<f32>(), &[12.0, 13.0, 14.0]);
}
