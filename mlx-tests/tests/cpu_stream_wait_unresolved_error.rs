//! A GPU error that a CPU-stream wait could not see yet must still fail the eval.
//!
//! A successful Metal command buffer signals its events BEFORE its completion handler runs, and
//! that handler is where an earlier failed buffer's sticky error is put on them. A CPU-stream wait
//! may not wait for the handler (a later GPU wait in that buffer can depend on the CPU stream), so
//! when an eval's intermediate buffer failed, a CPU op could consume a GPU result, see no error,
//! and the eval of the CPU-stream output returned corrupted arrays as `Ok`.
//!
//! `mlx-sys/patches/cpu-stream-wait-error.patch` records such an unresolved wait as a dependency
//! of the CPU stream; the stream's next signal attaches it to the signaled event, and the host
//! wait on that event waits for the dependency's handler and throws its error.
//!
//! The ordering is forced through the debug-only hook
//! `mlx_pmetal_test_fail_next_intermediate_command_buffer_until_cpu_signal` (`!NDEBUG` builds):
//! the next buffer committed without a signal event fails, and its completion handler is held
//! until a CPU stream signals an event. Metal runs a queue's handlers in order, so the later
//! buffer carrying the GPU->CPU fence signal cannot be poisoned before the CPU op has run and the
//! CPU stream has signaled the eval's event.
//!
//! This file holds a single test on purpose: `MLX_MAX_OPS_PER_BUFFER` is read once, when the Metal
//! device is created, and must be set before any MLX call in the process.

use mlx_rs::{array, Array, StreamOrDevice};

extern "C" {
    fn mlx_pmetal_test_fail_next_intermediate_command_buffer_until_cpu_signal();
}

/// A chain of GPU adds spanning several command buffers, consumed by an add on the CPU stream.
/// Kept short so the in-flight buffers stay under the eval's active-task throttle.
fn gpu_chain_then_cpu() -> (Array, Array) {
    let mut gpu = array!([1.0f32, 2.0, 3.0]);
    for _ in 0..8 {
        gpu = gpu.add(&array!([1.0f32, 1.0, 1.0])).expect("gpu add");
    }
    let cpu = gpu
        .add_device(&array!([10.0f32, 10.0, 10.0]), StreamOrDevice::cpu())
        .expect("cpu add");
    (gpu, cpu)
}

#[test]
fn gpu_error_unresolved_at_the_cpu_stream_wait_fails_the_eval() {
    // SAFETY: set before the first MLX call; this binary runs no other test.
    unsafe { std::env::set_var("MLX_MAX_OPS_PER_BUFFER", "2") };

    // 1. Control: the same cross-stream eval succeeds and computes the right values.
    let (_, clean) = gpu_chain_then_cpu();
    mlx_rs::transforms::eval([&clean]).expect("a clean GPU->CPU eval should succeed");
    assert_eq!(clean.as_slice::<f32>(), &[19.0, 20.0, 21.0]);

    // 2. Fail an intermediate GPU buffer whose error is recorded only after the CPU stream has
    //    consumed the GPU result and signaled the eval's event.
    unsafe { mlx_pmetal_test_fail_next_intermediate_command_buffer_until_cpu_signal() };
    let (_, poisoned) = gpu_chain_then_cpu();
    let err = mlx_rs::transforms::eval([&poisoned])
        .expect_err("a failed GPU buffer must fail the eval of a CPU-stream output, not return Ok");
    assert!(
        err.what().contains("synthetic intermediate-buffer failure"),
        "unexpected error surfaced: {}",
        err.what()
    );

    // 3. Reported once: both streams recover, so later, unrelated work is not poisoned.
    let (gpu, cpu) = gpu_chain_then_cpu();
    mlx_rs::transforms::eval([&cpu, &gpu]).expect("eval should recover once the error is reported");
    assert_eq!(gpu.as_slice::<f32>(), &[9.0, 10.0, 11.0]);
    assert_eq!(cpu.as_slice::<f32>(), &[19.0, 20.0, 21.0]);
}
