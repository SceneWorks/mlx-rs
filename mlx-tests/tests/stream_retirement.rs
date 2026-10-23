//! Rust/C/native retirement ABI smoke and selected-stream device metadata tests.
//! GPU metadata cases require the existing hosted Metal validation environment.
//! Real retained-owner/destructor and Metal tests are in the patched native test suite.
use mlx_rs::{
    array, task_local_default_stream, with_new_default_stream, Device, DeviceType, Stream,
    StreamOrDevice,
};

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

#[test]
fn selected_stream_device_metadata_preserves_default_and_custom_cpu_streams() {
    let cpu = Device::cpu();
    let default = Stream::cpu();
    let custom = Stream::new_with_device(&cpu);
    let previous_task_local = task_local_default_stream();
    assert_ne!(custom, default);
    for stream in [&default, &custom] {
        let index = stream.get_index().expect("stream index");
        let device = stream.get_device().expect("selected CPU stream device");
        assert_eq!(device, cpu);
        assert!(matches!(
            device.get_type().expect("device type"),
            DeviceType::Cpu
        ));
        assert_eq!(device.get_index().expect("device index"), 0);
        assert_eq!(stream.get_index().expect("unchanged stream index"), index);
    }
    with_new_default_stream(custom.clone(), || {
        let selected = Stream::task_local_or_default();
        assert_eq!(selected, custom);
        assert_eq!(selected.get_device().expect("task-local CPU device"), cpu);
        assert_eq!(Stream::cpu(), default);
    });
    assert_eq!(task_local_default_stream(), previous_task_local);
    assert_eq!(Stream::cpu(), default);
}

#[test]
#[ignore = "requires Metal; executed explicitly by validate_native_retirement.sh"]
fn selected_stream_device_metadata_preserves_gpu_streams_under_cpu_task_local() {
    let gpu = Device::gpu();
    let default_gpu = Stream::gpu();
    let custom_gpu = Stream::new_with_device(&gpu);
    let default_cpu = Stream::cpu();
    let previous_task_local = task_local_default_stream();
    assert_ne!(custom_gpu, default_gpu);
    with_new_default_stream(default_cpu.clone(), || {
        let selected = Stream::task_local_or_default();
        assert_eq!(selected, default_cpu);
        assert!(matches!(
            selected
                .get_device()
                .expect("task-local device")
                .get_type()
                .expect("type"),
            DeviceType::Cpu
        ));
        for stream in [&default_gpu, &custom_gpu] {
            let index = stream.get_index().expect("GPU stream index");
            let device = stream.get_device().expect("selected GPU stream device");
            assert_eq!(device, gpu);
            assert!(matches!(
                device.get_type().expect("device type"),
                DeviceType::Gpu
            ));
            assert_eq!(device.get_index().expect("device index"), 0);
            assert_eq!(
                stream.get_index().expect("unchanged GPU stream index"),
                index
            );
        }
        assert_eq!(StreamOrDevice::gpu().as_ref(), &default_gpu);
        assert_eq!(Stream::task_local_or_default(), default_cpu);
    });
    assert_eq!(task_local_default_stream(), previous_task_local);
    assert_eq!(Stream::gpu(), default_gpu);
    assert_eq!(Stream::cpu(), default_cpu);
}
