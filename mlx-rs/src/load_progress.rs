//! Reader-owned progress for lazy safetensors I/O.
//!
//! A callback is attached to one file reader, retained by its lazy arrays and queued reads.
//! It can run on MLX I/O threads after the loading call returns. Callbacks must return promptly,
//! must not invoke MLX, and should coalesce or enqueue work rather than perform network I/O.

use std::ffi::c_void;
use std::sync::Arc;

/// Actual file-reader activity, including the header and lazy tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadProgressEvent {
    /// At least one file read has begun. No bytes have been credited yet.
    Started {
        /// Size of the opened file in bytes.
        total_bytes: u64,
    },
    /// Bytes successfully returned by `read`/`pread`, never the mapped or requested size.
    BytesRead {
        /// Bytes successfully returned by this read.
        bytes: u64,
        /// Size of the opened file in bytes.
        total_bytes: u64,
    },
    /// All currently scheduled reads are idle, including after a read error.
    /// Further lazy tensor reads may begin later.
    Idle,
}

/// A per-file callback. Panics are contained at the FFI boundary and never unwind into C++.
#[derive(Clone)]
pub struct LoadProgress {
    callback: Arc<dyn Fn(LoadProgressEvent) + Send + Sync>,
    capture: Option<Arc<dyn Fn() -> LoadProgress + Send + Sync>>,
}

impl std::fmt::Debug for LoadProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadProgress").finish_non_exhaustive()
    }
}

impl LoadProgress {
    /// Create a thread-safe callback owned by one lazy file-reader lifetime.
    pub fn new(callback: impl Fn(LoadProgressEvent) + Send + Sync + 'static) -> Self {
        Self {
            callback: Arc::new(callback),
            capture: None,
        }
    }

    /// Capture a fresh callback on the evaluating caller's thread before each lazy Load is queued.
    /// The returned callback is retained only by that immutable read token. A panicking factory
    /// falls back to the original callback. With no factory, reads retain the original callback.
    pub fn with_evaluation_capture(
        mut self,
        capture: impl Fn() -> LoadProgress + Send + Sync + 'static,
    ) -> Self {
        self.capture = Some(Arc::new(capture));
        self
    }

    /// Deliver an event from a custom reader, containing callback panics just like native reads.
    pub fn report(&self, event: LoadProgressEvent) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.callback)(event)));
    }

    pub(crate) fn into_context(self) -> *mut c_void {
        Box::into_raw(Box::new(self)).cast()
    }
}

pub(crate) unsafe extern "C" fn report(
    context: *mut c_void,
    bytes: u64,
    total_bytes: u64,
    event: i32,
) {
    // SAFETY: the native reader owns this Box until its release callback; active reads retain it.
    let callback = unsafe { &*context.cast::<LoadProgress>() };
    let event = match event {
        0 => LoadProgressEvent::Started { total_bytes },
        1 => LoadProgressEvent::BytesRead { bytes, total_bytes },
        2 => LoadProgressEvent::Idle,
        _ => return,
    };
    callback.report(event);
}

pub(crate) unsafe extern "C" fn release(context: *mut c_void) {
    // A user's closure can contain a panicking Drop, which must not cross FFI either.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: native ownership invokes release exactly once, after the last reader reference.
        drop(unsafe { Box::from_raw(context.cast::<LoadProgress>()) });
    }));
}

pub(crate) unsafe extern "C" fn capture(context: *mut c_void) -> *mut c_void {
    // SAFETY: capture runs synchronously while the native reader retains its original context.
    let original = unsafe { &*context.cast::<LoadProgress>() };
    let callback = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        original
            .capture
            .as_ref()
            .map(|capture| capture())
            .unwrap_or_else(|| original.clone())
    }))
    .unwrap_or_else(|_| original.clone());
    callback.into_context()
}
