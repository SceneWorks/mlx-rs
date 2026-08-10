use std::process::Command;

#[test]
fn retained_main_thread_handle_exits_without_touching_a_destroyed_compiler_cache() {
    let output = Command::new(env!("CARGO_BIN_EXE_retained_compile_teardown_probe"))
        .output()
        .expect("run retained compile teardown probe");
    assert!(
        output.status.success(),
        "probe exited as {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "retained compile teardown clean"
    );
}
