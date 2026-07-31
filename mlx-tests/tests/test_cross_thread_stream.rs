//! sc-12937 regression tests: streams must be usable across OS threads.
//!
//! MLX registers each stream's command encoder in a thread_local map, so
//! without the pmetal `thread-shared-streams.patch` a graph built on one
//! thread (whose nodes carry that thread's default stream) fails to evaluate
//! on any other thread with:
//!
//!   "There is no Stream(gpu, N) in current thread."
//!
//! Rust makes this cross-thread shape unavoidable: `Array` is `Send`, the
//! test harness runs every `#[test]` on its own spawned thread, and MLX's
//! global random key state is shared across threads by design. These tests
//! pin the fixed behavior deterministically (the original 20 failures were
//! test-order dependent).

use std::sync::Mutex;

use mlx_rs::{ops::ones, Array};

/// MLX evaluation is not thread safe (CI runs the whole suite with
/// `--test-threads=1` for the same reason). These tests intentionally move
/// graphs across threads and share the global random key, so two of them
/// evaluating concurrently can race the same stream's command encoder and
/// trip a Metal assertion. Serialize them so a plain `cargo test` (parallel
/// libtest threads) stays safe.
static SERIAL: Mutex<()> = Mutex::new(());

/// A lazy graph built on a spawned thread must evaluate on this thread.
///
/// This is the exact failure shape of sc-12937: the spawned thread's default
/// stream is created in that thread, the unevaluated nodes reference it, and
/// eval happens on a different thread. RED without the patch.
#[test]
fn test_eval_graph_built_on_other_thread() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let lazy = std::thread::spawn(|| {
        let x = mlx_rs::random::uniform::<_, f32>(1.0, 2.0, &[4, 4], None).unwrap();
        // Keep it lazy: build ops but do not evaluate on the building thread.
        x.add(&x).unwrap().sum(None).unwrap()
    })
    .join()
    .unwrap();

    // Unpatched, this eval throws "There is no Stream(gpu, N) in current thread."
    lazy.eval().unwrap();
    assert!(lazy.item::<f32>() > 0.0);
}

/// The reverse direction: a lazy graph built on THIS thread must evaluate on
/// a spawned thread. RED without the patch.
#[test]
fn test_eval_graph_on_spawned_thread() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ones::<f32>(&[2, 2]).unwrap();
    // Lazy graph whose nodes carry this thread's default stream.
    let lazy = a.add(&b).unwrap().sum(None).unwrap();

    let out = std::thread::spawn(move || {
        // Unpatched, this throws "There is no Stream(gpu, N) in current thread."
        lazy.eval().unwrap();
        lazy.item::<f32>()
    })
    .join()
    .unwrap();

    assert_eq!(out, 14.0);
}

/// MLX's global random key state must remain usable after another thread
/// advanced it (the KeySequence split nodes carry the other thread's stream).
/// RED without the patch.
#[test]
fn test_global_random_state_across_threads() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    std::thread::spawn(|| {
        // Advance the global random key on a different thread, leaving the
        // next split of the key state referencing that thread's stream.
        let _ = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 2], None).unwrap();
    })
    .join()
    .unwrap();

    let x = mlx_rs::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();
    // Unpatched, this eval throws: the split of the global key references the
    // spawned thread's stream.
    let mean = x.mean(None).unwrap();
    mean.eval().unwrap();
    let mean = mean.item::<f32>();
    assert!((1.0..2.0).contains(&mean), "uniform(1,2) mean was {mean}");
}

/// Arrays already evaluated on one thread must be usable as graph inputs from
/// another thread (Array is Send; pmetal moves arrays between tokio workers).
/// Guard rather than red/green pin: the graph nodes here are all created on
/// the evaluating thread, so this passes even unpatched — it guards the
/// evaluated-data path against future stream/thread coupling.
#[test]
fn test_evaluated_array_used_from_other_thread() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    a.eval().unwrap();

    let out = std::thread::spawn(move || {
        let b = ones::<f32>(&[2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        c.eval().unwrap();
        c.sum(None).unwrap().item::<f32>()
    })
    .join()
    .unwrap();

    assert_eq!(out, 14.0);
}

/// sc-12959: the default stream is process-global — every thread sees the
/// same stream for the default device (upstream: one per thread). This is
/// what caps encoder/queue count at O(devices) instead of O(threads that
/// ever touched MLX). RED without the thread-safe-eval patch (per-thread
/// defaults produce distinct stream indices).
#[test]
fn test_default_stream_shared_across_threads() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let here = mlx_rs::Stream::new().get_index().unwrap();
    let there = std::thread::spawn(|| mlx_rs::Stream::new().get_index().unwrap())
        .join()
        .unwrap();
    assert_eq!(
        here, there,
        "default stream must be process-global (sc-12959), got {here} vs {there}"
    );
}

/// sc-12959: concurrent evaluation from many threads must be safe — the
/// process-global eval lock serializes encoding, so this must neither SIGABRT
/// ("A command encoder is already encoding") nor corrupt results. With the
/// process-global default stream every thread contends on the SAME stream,
/// making this the worst case. Deterministic assertions per thread.
#[test]
fn test_concurrent_eval_from_many_threads() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                let scale = (i + 1) as f32;
                for _ in 0..25 {
                    let a = Array::from_slice(&[scale, scale, scale, scale], &[2, 2]);
                    let b = ones::<f32>(&[2, 2]).unwrap();
                    let sum = a.add(&b).unwrap().sum(None).unwrap();
                    sum.eval().unwrap();
                    assert_eq!(sum.item::<f32>(), 4.0 * (scale + 1.0));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
