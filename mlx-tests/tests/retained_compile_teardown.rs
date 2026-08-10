use std::os::unix::process::ExitStatusExt;
use std::process::Command;

#[test]
fn retained_main_and_spawned_worker_handles_survive_tls_teardown() {
    for mode in ["main", "worker"] {
        let output = Command::new(env!("CARGO_BIN_EXE_retained_compile_teardown_probe"))
            .arg(mode)
            .output()
            .expect("run retained compile teardown probe");
        assert!(
            output.status.success(),
            "{mode} probe exited as {:?} (signal {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            output.status.signal(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            format!("retained compile {mode} teardown clean")
        );
    }
}
