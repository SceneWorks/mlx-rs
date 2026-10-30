//! I/O-only tests: CPU Load primitives copy safetensors bytes; no learned operations run.
use mlx_rs::load_progress::{LoadProgress, LoadProgressEvent};
use mlx_rs::{Array, Stream};
use std::io::Write;
use std::sync::{Arc, Mutex};

fn fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("weights.safetensors");
    let header = br#"{"a":{"dtype":"U8","shape":[65536],"data_offsets":[0,65536]},"b":{"dtype":"U8","shape":[65536],"data_offsets":[65536,131072]}}"#;
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&vec![17; 131072]).unwrap();
    path
}

#[test]
fn lazy_cpu_reads_report_exact_header_and_payload_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let total = std::fs::metadata(&path).unwrap().len();
    let events = Arc::new(Mutex::new(Vec::new()));
    let output = events.clone();
    let owner = Arc::new(());
    let alive = Arc::downgrade(&owner);
    let progress = LoadProgress::new(move |event| {
        let _retain = &owner;
        output.lock().unwrap().push(event);
    });
    let arrays =
        Array::load_safetensors_with_progress_device(&path, progress, Stream::cpu()).unwrap();
    let bytes = || {
        events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                LoadProgressEvent::BytesRead { bytes, total_bytes } => {
                    assert_eq!(*total_bytes, total);
                    Some(bytes)
                }
                _ => None,
            })
            .sum::<u64>()
    };
    assert_eq!(bytes(), total - 131072, "only header reads happen at open");
    assert_eq!(
        events.lock().unwrap().last(),
        Some(&LoadProgressEvent::Idle)
    );
    assert!(
        alive.upgrade().is_some(),
        "lazy arrays retain callback context"
    );
    mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
    assert_eq!(bytes(), total);
    mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
    assert_eq!(
        bytes(),
        total,
        "resident arrays do not fabricate another read"
    );
    assert_eq!(
        events.lock().unwrap().last(),
        Some(&LoadProgressEvent::Idle)
    );
    drop(arrays);
    assert!(alive.upgrade().is_none());
}

#[test]
fn dropped_lazy_arrays_release_context_without_reading_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let owner = Arc::new(());
    let alive = Arc::downgrade(&owner);
    let arrays = Array::load_safetensors_with_progress_device(
        path,
        LoadProgress::new(move |_| {
            let _retain = &owner;
        }),
        Stream::cpu(),
    )
    .unwrap();
    drop(arrays);
    assert!(alive.upgrade().is_none());
}

#[test]
fn callback_panics_cannot_unwind_into_cpp_and_read_errors_clear_activity() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let events = Arc::new(Mutex::new(Vec::new()));
    let output = events.clone();
    let arrays = Array::load_safetensors_with_progress_device(
        &path,
        LoadProgress::new(move |event| {
            output.lock().unwrap().push(event);
            panic!("contained callback panic");
        }),
        Stream::cpu(),
    )
    .unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(8)
        .unwrap();
    assert!(mlx_rs::transforms::eval_pending_loads(arrays.values()).is_err());
    assert_eq!(
        events.lock().unwrap().last(),
        Some(&LoadProgressEvent::Idle)
    );
}

#[test]
fn independent_readers_keep_their_own_callbacks() {
    let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
    let counters: Vec<_> = (0..2).map(|_| Arc::new(Mutex::new(0u64))).collect();
    let arrays: Vec<_> = dirs
        .iter()
        .zip(&counters)
        .map(|(dir, count)| {
            let path = fixture(dir.path());
            let output = count.clone();
            Array::load_safetensors_with_progress_device(
                path,
                LoadProgress::new(move |event| {
                    if let LoadProgressEvent::BytesRead { bytes, .. } = event {
                        *output.lock().unwrap() += bytes;
                    }
                }),
                Stream::cpu(),
            )
            .unwrap()
        })
        .collect();
    let second_header = *counters[1].lock().unwrap();
    mlx_rs::transforms::eval_pending_loads(arrays[0].values()).unwrap();
    assert_eq!(
        *counters[0].lock().unwrap(),
        std::fs::metadata(dirs[0].path().join("weights.safetensors"))
            .unwrap()
            .len()
    );
    assert_eq!(*counters[1].lock().unwrap(), second_header);
    mlx_rs::transforms::eval_pending_loads(arrays[1].values()).unwrap();
    assert_eq!(*counters[0].lock().unwrap(), *counters[1].lock().unwrap());
}

#[test]
fn parallel_chunk_reads_report_each_successful_byte_once() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.safetensors");
    let payload = (1u64 << 25) + 1;
    let header =
        format!(r#"{{"w":{{"dtype":"U8","shape":[{payload}],"data_offsets":[0,{payload}]}}}}"#);
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    let total = 8 + header.len() as u64 + payload;
    file.set_len(total).unwrap();
    let count = Arc::new(AtomicU64::new(0));
    let output = count.clone();
    let arrays = Array::load_safetensors_with_progress_device(
        &path,
        LoadProgress::new(move |event| {
            if let LoadProgressEvent::BytesRead { bytes, .. } = event {
                output.fetch_add(bytes, Ordering::Relaxed);
            }
        }),
        Stream::cpu(),
    )
    .unwrap();
    mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
    assert_eq!(count.load(Ordering::Relaxed), total);
}

#[test]
fn invalid_header_releases_callback_context_on_open_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.safetensors");
    std::fs::write(&path, [0; 8]).unwrap();
    let owner = Arc::new(());
    let alive = Arc::downgrade(&owner);
    let progress = LoadProgress::new(move |_| {
        let _retain = &owner;
    });
    assert!(Array::load_safetensors_with_progress_device(path, progress, Stream::cpu()).is_err());
    assert!(alive.upgrade().is_none());
}

#[test]
fn evaluation_capture_runs_on_caller_and_async_reads_keep_immutable_token() {
    use std::sync::mpsc;
    thread_local! { static REQUEST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let (started, start_received) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let released = Arc::new(Mutex::new(released));
    let events = Arc::new(Mutex::new(Vec::new()));
    let output = events.clone();
    let captured_thread = std::thread::current().id();
    let progress = LoadProgress::new(|_| {}).with_evaluation_capture(move || {
        assert_eq!(std::thread::current().id(), captured_thread);
        let request = REQUEST.with(|v| v.get());
        let output = output.clone();
        let started = started.clone();
        let released = released.clone();
        LoadProgress::new(move |event| {
            if let LoadProgressEvent::Started { .. } = event {
                started.send(()).unwrap();
                // Test-only rendezvous forces the actual read to remain pending while caller TLS changes.
                released.lock().unwrap().recv().unwrap();
            }
            if let LoadProgressEvent::BytesRead { bytes, .. } = event {
                output.lock().unwrap().push((request, bytes));
            }
        })
    });
    let arrays =
        Array::load_safetensors_with_progress_device(&path, progress, Stream::cpu()).unwrap();
    REQUEST.with(|v| v.set(11));
    mlx_rs::transforms::async_eval(arrays.values()).unwrap();
    start_received
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("scheduled reader must report Started");
    REQUEST.with(|v| v.set(22));
    release.send(()).unwrap();
    release.send(()).unwrap();
    mlx_rs::transforms::eval(arrays.values()).unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events.iter().map(|(_, bytes)| bytes).sum::<u64>(), 131072);
    assert!(events.iter().all(|(request, _)| *request == 11));
}

#[test]
fn every_fp8_storage_dtype_remains_a_lazy_bit_exact_byte_view() {
    let dir = tempfile::tempdir().unwrap();
    for dtype in ["F8_E4M3", "F8_E5M2", "F8_E8M0"] {
        let path = dir.path().join(format!("{dtype}.safetensors"));
        let header = format!(r#"{{"w":{{"dtype":"{dtype}","shape":[4],"data_offsets":[0,4]}}}}"#);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&[0, 127, 128, 255]).unwrap();
        let arrays = Array::load_safetensors_device(&path, Stream::cpu()).unwrap();
        mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
        assert_eq!(arrays["w"].dtype(), mlx_rs::Dtype::Uint8);
        assert_eq!(arrays["w"].shape(), &[4]);
        assert_eq!(arrays["w"].as_slice::<u8>(), &[0, 127, 128, 255]);
    }
}
