//! Failed lazy reads must remain errors through CPU/GPU and asynchronous graphs.
use mlx_rs::{transforms, Array, StreamOrDevice};
use std::collections::HashMap;

fn write_weights(path: &std::path::Path) {
    let arrays = HashMap::from([(
        "w".to_string(),
        Array::from_slice(&[1f32, 2., 3., 4.], &[4]),
    )]);
    Array::save_safetensors(&arrays, None, path).unwrap();
}

#[test]
fn failed_loads_survive_async_dependencies_and_repeated_waits() {
    for device in [StreamOrDevice::cpu(), StreamOrDevice::gpu()] {
        for asynchronous in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("truncated.safetensors");
            write_weights(&path);
            let w = Array::load_safetensors(&path).unwrap().remove("w").unwrap();
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(0)
                .unwrap();
            let out = w.square_device(&device).unwrap();
            if asynchronous {
                transforms::async_eval([&out]).unwrap();
            }
            let derived = out.sum_device(false, &device).unwrap();
            for _ in 0..2 {
                let error = derived
                    .try_item::<f32>()
                    .expect_err("failed reads must never become usable zero arrays")
                    .to_string();
                assert!(error.contains("truncated.safetensors"), "{error}");
                assert!(
                    error.contains("offset=") && error.contains("unexpected EOF"),
                    "{error}"
                );
            }
            assert!(
                out.eval().is_err(),
                "the original result must remain poisoned too"
            );
            // No global error slot: a subsequent independent load must work.
            write_weights(&path);
            let good = Array::load_safetensors(&path).unwrap().remove("w").unwrap();
            assert_eq!(
                good.sum_device(false, &device)
                    .unwrap()
                    .try_item::<f32>()
                    .unwrap(),
                10.
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn read_fault_child() {
    let Ok(mode) = std::env::var("MLX_TEST_READ_FAULT") else {
        return;
    };
    // Cross the reader's 32 MiB batching boundary and include a partial tail.
    let values: Vec<f32> = (0..9_000_001).map(|i| (i % 1021) as f32).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fault.safetensors");
    Array::save_safetensors(
        &HashMap::from([(
            "w".to_string(),
            Array::from_slice(&values, &[values.len() as i32]),
        )]),
        None,
        &path,
    )
    .unwrap();
    for device in [StreamOrDevice::cpu(), StreamOrDevice::gpu()] {
        for asynchronous in [false, true] {
            let w = Array::load_safetensors(&path).unwrap().remove("w").unwrap();
            let out = w.add_device(Array::from_f32(1.), &device).unwrap();
            if asynchronous {
                transforms::async_eval([&out]).unwrap();
            }
            let result = out.eval();
            if mode == "efault" {
                let error = result.expect_err("EFAULT must reach the host").to_string();
                assert!(
                    error.contains("fault.safetensors")
                        && error.contains("offset=")
                        && error.contains("errno=14"),
                    "{error}"
                );
                assert!(out.eval().is_err());
                assert!(out
                    .sum_device(false, &device)
                    .unwrap()
                    .try_item::<f32>()
                    .is_err());
            } else {
                result.unwrap();
                let actual = out.as_slice::<f32>();
                assert_eq!(actual.len(), values.len());
                for (i, (&a, &v)) in actual.iter().zip(&values).enumerate() {
                    assert_eq!(a, v + 1., "wrong bytes at {i} under {mode}");
                }
            }
            let good_path = dir.path().join("healthy.safetensors");
            write_weights(&good_path);
            let good = Array::load_safetensors(&good_path)
                .unwrap()
                .remove("w")
                .unwrap();
            assert_eq!(
                good.sum_device(false, &device)
                    .unwrap()
                    .try_item::<f32>()
                    .unwrap(),
                10.
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn positioned_read_faults_are_recoverable_and_partial_reads_are_exact() {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let dir = tempfile::tempdir().unwrap();
    let dylib = dir.path().join("read_fault.dylib");
    assert!(Command::new("clang")
        .args(["-dynamiclib", "-O2", "-std=c11"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/read_fault.c"))
        .arg("-o")
        .arg(&dylib)
        .status()
        .unwrap()
        .success());
    for mode in ["efault", "eintr", "short"] {
        let log = dir.path().join(format!("{mode}.log"));
        let output = std::fs::File::create(&log).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "read_fault_child", "--nocapture"])
            .env("DYLD_INSERT_LIBRARIES", &dylib)
            .env("MLX_TEST_READ_FAULT", mode)
            .stdout(Stdio::from(output.try_clone().unwrap()))
            .stderr(Stdio::from(output))
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!(
                    "{mode} child hung: {}",
                    std::fs::read_to_string(&log).unwrap()
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let output = std::fs::read_to_string(log).unwrap();
        assert!(
            status.success(),
            "{mode} aborted/failed: {status}: {output}"
        );
        assert!(
            output.contains("MLX_TEST_READ_FAULT injected"),
            "interposer was not exercised: {output}"
        );
    }
}

#[test]
fn failed_batch_does_not_poison_independent_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let bad_path = dir.path().join("truncated.safetensors");
    let good_path = dir.path().join("healthy.safetensors");
    for asynchronous in [false, true] {
        write_weights(&bad_path);
        write_weights(&good_path);
        let bad = Array::load_safetensors(&bad_path)
            .unwrap()
            .remove("w")
            .unwrap()
            .square()
            .unwrap();
        let good = Array::load_safetensors(&good_path)
            .unwrap()
            .remove("w")
            .unwrap()
            .square()
            .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&bad_path)
            .unwrap()
            .set_len(0)
            .unwrap();
        if asynchronous {
            transforms::async_eval([&bad, &good]).unwrap();
        } else {
            assert!(transforms::eval([&bad, &good]).is_err());
        }
        assert!(bad.eval().is_err());
        assert_eq!(good.sum(false).unwrap().try_item::<f32>().unwrap(), 30.);
        assert!(bad.add(&good).unwrap().eval().is_err());
    }
}
