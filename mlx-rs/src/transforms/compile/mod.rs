//! Compilation of functions.
//!
//! See also [MLX python
//! documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
//!
//! MLX has a [`compile()`] function transformation which compiles computation
//! graphs. Function compilation results in smaller graphs by merging common
//! work and fusing certain operations. In many cases this can lead to big
//! improvements in run-time and memory use.
//!
//! Getting started with compile() is simple, but there are some edge cases that
//! are good to be aware of for more complex graphs and advanced usage.
//!
//! **WARN**: Because function transforms including compilation works on the
//! computation graph, the user must ensure that all `Array`s are passed as
//! inputs to the function/closure. Closures with captured `Array`s may not work
//! as expected and may lead to undefined behavior.
//!
//! # Basic usage
//!
//! ```rust
//! use mlx_rs::{Array, array, transforms::compile::compile, error::Exception};
//!
//! let fun = |(x, y): (&Array, &Array)| -> Result<Array, Exception> {
//!    mlx_rs::exp!(x.negative()?)?.add(y)
//! };
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! // Regular call, no compilation
//! let result = fun((&x, &y)).unwrap();
//! // Prints: array(2.36788, dtype=float32)
//! println!("{:?}", result);
//!
//! // Compile the function
//! let mut compiled_fun = compile(fun, None);
//! let result = compiled_fun((&x, &y)).unwrap();
//! // Prints: array(2.36788, dtype=float32)
//! println!("{:?}", result);
//! ```
//!
//! The output of both the regular function and the compiled function is the
//! same up to numerical precision.
//!
//! The first time you call a compiled function, MLX will build the compute
//! graph, optimize it, and generate and compile code. This can be relatively
//! slow. However, MLX will cache a retained compiled function, so calling that
//! handle multiple times will not initiate a new compilation. This means you
//! should typically retain compiled functions that you plan to use more than once.
//!
//! ```rust
//! use mlx_rs::{Array, array, transforms::compile::compile_retained};
//!
//! let fun = |(x, y): (&Array, &Array)| {
//!    mlx_rs::exp!(x.negative()?)?.add(y)
//! };
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! let mut compiled_fun = compile_retained(fun, None);
//!
//! // Compiled here
//! let result = compiled_fun.call_mut((&x, &y)).unwrap();
//!
//! // Not compiled again
//! let result = compiled_fun.call_mut((&x, &y)).unwrap();
//! ```
//!
//! There are some important cases to be aware of that can cause a function to
//! be recompiled:
//!
//! - Changing the shape or number of dimensions
//! - Changing the type of any of the inputs
//! - Changing the number of inputs to the function
//!
//! In certain cases only some of the compilation stack will be rerun (for
//! example when changing the shapes) and in other cases the full compilation
//! stack will be rerun (for example when changing the types). In general you
//! should avoid compiling functions too frequently.
//!
//! Another idiom to watch out for is compiling functions which get created and
//! destroyed frequently. This can happen, for example, when compiling an
//! closure in a loop.
//!
//! # Pure Functions
//!
//! Compiled functions are intended to be pure; that is they should not have
//! side effects. For example:
//!
//! ```rust,ignore
//! use mlx_rs::{Array, array, transforms::compile::compile};
//!
//! let mut c = array!(0.5);
//!
//! let fun = |(x, y): (&Array, &Array)| {
//!     let z = (x + y) * c;
//!     mlx_rs::exp!(z)
//! };
//!
//! let mut compiled = compile(fun, None);
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! // This may lead to undefined behavior
//! let result = compiled((&x, &y)).unwrap();
//! println!("{:?}", result);
//! ```
//!
//! Use [`compile_with_state()`] to compile functions that have side effects and
//! pass the state as an mutable reference.
//!
//! ```rust
//! use mlx_rs::{Array, array, transforms::compile::compile_with_state};
//! let mut state = vec![];
//!
//! let fun = |state: &mut Vec<Array>, (x, y): (&Array, &Array)| {
//!     let z = x + y;
//!     let result = mlx_rs::exp!(&z);
//!     state.push(z);
//!     result
//! };
//!
//! let x = array!(1.0);
//! let y = array!(2.0);
//!
//! let mut compiled = compile_with_state(fun, None);
//! let result = compiled(&mut state, (&x, &y)).unwrap();
//! println!("{:?}", result);
//! // println!("{:?}", state); // TODO: this currently doesn't work somehow
//! ```
//!
//! This is particularly useful for compiling a function which includes an
//! update to a container of arrays, as is commonly done when training the
//! parameters of a [`crate::module::Module`].
//!
//! See mlx-rs/mlx-tests/tests/test_compile_with_state.rs for more examples.
//!

use std::{
    rc::Rc,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use super::{Closure, Guarded, VectorArray};
use crate::Array;

#[allow(clippy::module_inception)]
mod compile;
mod compile_with_state;

pub use compile::*;
pub use compile_with_state::*;

static NEXT_COMPILE_ID: AtomicUsize = AtomicUsize::new(1);
static CACHE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Shared ownership of one backend compiler-cache identity.
///
/// `Compiled` is cloneable, so erasing from the backend in `CompiledState::drop` let the first
/// clone invalidate every sibling. Keeping the identity behind an `Rc` makes the erase happen
/// exactly once, after the final Rust handle is gone. The backend cache is thread-local, so the
/// lease deliberately remains bound to the thread that created it.
#[derive(Debug)]
struct CompileLease {
    id: usize,
    cache_token: *mut std::ffi::c_void,
}

impl Drop for CompileLease {
    fn drop(&mut self) {
        unsafe {
            // The token outlives both the MLX compiler cache and this lease. It erases immediately
            // during an ordinary drop, but only releases its retained token reference when a Rust
            // TLS destructor runs after MLX has already destroyed the owning thread's cache.
            let _ = mlx_sys::mlx_detail_compile_erase_with_cache_token(self.cache_token, self.id);
        }
    }
}

fn new_compile_lease() -> Rc<CompileLease> {
    let id = NEXT_COMPILE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("exhausted process-local MLX compile identities");
    let cache_token = unsafe { mlx_sys::mlx_detail_compile_acquire_cache_token() };
    assert!(
        !cache_token.is_null(),
        "MLX failed to acquire the current thread's compiler-cache lifetime token"
    );
    Rc::new(CompileLease { id, cache_token })
}

/// Globally enable the compilation of functions.
///
/// Default is enabled.
pub fn enable_compile() {
    unsafe {
        mlx_sys::mlx_enable_compile();
    }
}

/// Globally disable the compilation of functions.
///
/// Default is enabled.
pub fn disable_compile() {
    unsafe {
        mlx_sys::mlx_disable_compile();
    }
}

/// Clear the memory cache.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_detail_compile_clear_cache();
    }
    CACHE_GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// Process-local generation of MLX's compiler cache.
///
/// Retained-callable diagnostics use this to distinguish the first invocation after an explicit
/// [`clear_cache`] from a true cache hit. It is monotonic for the process lifetime.
pub fn cache_generation() -> u64 {
    CACHE_GENERATION.load(Ordering::Acquire)
}

/// Initialize MLX's per-thread compiler cache and its retained-handle lifetime token.
///
/// Rust and C++ thread-local destructors are not guaranteed to share one cross-language ordering.
/// The native lifetime token therefore remains valid until every Rust lease has dropped and marks
/// the cache unavailable before C++ destroys it. Calling this function before accessing a
/// downstream TLS cache remains recommended and is harmless; repeated calls neither clear nor
/// otherwise mutate compiled entries.
pub fn prepare_retained_compilation_thread() {
    unsafe {
        mlx_sys::mlx_detail_compile_initialize_cache();
    }
}

/// A compiled function that can be called.
#[derive(Debug, Clone)]
pub struct Compiled<F, G> {
    f_marker: std::marker::PhantomData<F>,
    state: CompiledState<G>,
}

#[derive(Debug, Clone)]
struct CompiledState<F> {
    f: F,
    shapeless: bool,
    lease: Rc<CompileLease>,
}

fn update_by_replace_with_ref_to_new_array(src: &mut Array, new_array: &Array) {
    debug_assert_eq!(src.shape(), new_array.shape());
    unsafe {
        mlx_sys::mlx_array_set(&mut src.as_ptr() as *mut _, new_array.as_ptr());
    }
}
