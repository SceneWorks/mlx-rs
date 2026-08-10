use std::cell::RefCell;

use mlx_rs::{
    array,
    ops::multiply,
    transforms::compile::{compile_retained, prepare_retained_compilation_thread, RetainedUnary},
    Array,
};

thread_local! {
    // Intentionally survives `main`: this is the production lifecycle that used to call
    // compile_erase after MLX's C++ compiler-cache static had already been destroyed.
    static RETAINED: RefCell<Option<Box<dyn RetainedUnary>>> = const { RefCell::new(None) };
}

fn main() {
    let x = array!([2.0f32, 3.0]);
    prepare_retained_compilation_thread();
    RETAINED.with(|slot| {
        let mut slot = slot.borrow_mut();
        let compiled =
            slot.get_or_insert_with(|| compile_retained(|arg: &Array| multiply(arg, arg), true));
        assert_eq!(compiled.call_mut(&x).unwrap(), array!([4.0f32, 9.0]));
        assert_eq!(compiled.call_mut(&x).unwrap(), array!([4.0f32, 9.0]));
    });
    println!("retained compile teardown clean");
}
