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

use mlx_rs::{ops::ones, Array};

/// A lazy graph built on a spawned thread must evaluate on this thread.
///
/// This is the exact failure shape of sc-12937: the spawned thread's default
/// stream is created in that thread, the unevaluated nodes reference it, and
/// eval happens on a different thread.
#[test]
fn test_eval_graph_built_on_other_thread() {
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

/// MLX's global random key state must remain usable after another thread
/// advanced it (the KeySequence split nodes carry the other thread's stream).
#[test]
fn test_global_random_state_across_threads() {
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

/// Arrays evaluated on one thread must be usable as graph inputs from another
/// thread (Array is Send; pmetal moves arrays between tokio workers).
#[test]
fn test_evaluated_array_used_from_other_thread() {
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
