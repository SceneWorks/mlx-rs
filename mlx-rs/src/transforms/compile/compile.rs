//! Compilation of functions.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::marker::PhantomData;

use crate::{error::Exception, Array};

use super::{
    new_compile_lease, prepare_retained_compilation_thread, Closure, Compiled, CompiledState,
    Guarded, VectorArray,
};

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(F::Args<'a>) -> Result<O, Exception>
where
    F: Compile<A, O, E> + 'static + Copy,
{
    let shapeless = shapeless.into().unwrap_or(false);
    move |args| {
        // NOTE: we have to place this here to avoid the lifetime issue. This convenience API owns a
        // one-call handle; use `compile_retained` when the backend graph must survive across calls.
        let mut compiled = f.compile(shapeless);
        compiled.call_mut(args)
    }
}

/// Returns a compiled function whose inner MLX compiled-state handle remains alive until the
/// returned callable is dropped.
///
/// This is the retained counterpart to [`compile`]. The convenience [`compile`] closure constructs
/// an inner [`Compiled`] handle for each invocation so it can erase the argument lifetime from its
/// public `FnMut` type. That is ergonomic for one-shot use, but dropping the inner handle also erases
/// MLX's compiled structure. `compile_retained` instead returns one of the object-safe retained call
/// interfaces ([`RetainedSlice`], [`RetainedUnary`], [`RetainedBinary`], or [`RetainedTernary`]),
/// whose `call_mut` method is lifetime-generic and keeps the inner handle alive across calls.
///
/// Both infallible and [`Exception`]-returning functions are supported for every input arity
/// advertised by [`compile`].
///
/// # Example
///
/// ```rust
/// use mlx_rs::{array, ops::add, transforms::compile::compile_retained, Array};
///
/// let function = |(x, y): (&Array, &Array)| add(x, y);
/// let mut compiled = compile_retained(function, true);
/// let x = array!(1.0f32);
/// let y = array!(2.0f32);
///
/// let first = compiled.call_mut((&x, &y)).unwrap();
/// let second = compiled.call_mut((&x, &y)).unwrap();
/// assert_eq!(first.item::<f32>(), second.item::<f32>());
/// ```
pub fn compile_retained<F, A, O, E>(f: F, shapeless: impl Into<Option<bool>>) -> F::Callable
where
    F: CompileRetained<A, O, E>,
{
    prepare_retained_compilation_thread();
    f.compile_retained(shapeless.into().unwrap_or(false))
}

/// Type-directed construction of a retained compiled callable.
///
/// Use the [`compile_retained`] free function rather than naming this trait directly. `A`, `O`, and
/// `E` mirror [`Compile`]'s input, output, and error markers and allow Rust to select the supported
/// arity and fallibility without overloads.
pub trait CompileRetained<A, O, E>: Sized {
    /// The object-safe callable returned for this input arity.
    type Callable;

    /// Construct the retained callable.
    fn compile_retained(self, shapeless: bool) -> Self::Callable;
}

/// A retained compiled function accepting a borrowed slice of arrays and returning multiple arrays.
pub trait RetainedSlice {
    /// Apply the retained compiled function.
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception>;
}

/// A retained compiled function accepting one array.
pub trait RetainedUnary {
    /// Apply the retained compiled function.
    fn call_mut(&mut self, arg: &Array) -> Result<Array, Exception>;
}

/// A retained compiled function accepting two arrays.
pub trait RetainedBinary {
    /// Apply the retained compiled function.
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception>;
}

/// A retained compiled function accepting three arrays.
pub trait RetainedTernary {
    /// Apply the retained compiled function.
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception>;
}

struct RetainedSliceImpl<F, G, E> {
    compiled: Compiled<F, G>,
    error: PhantomData<E>,
}

struct RetainedUnaryImpl<F, G, E> {
    compiled: Compiled<F, G>,
    error: PhantomData<E>,
}

struct RetainedBinaryImpl<F, G, E> {
    compiled: Compiled<F, G>,
    error: PhantomData<E>,
}

struct RetainedTernaryImpl<F, G, E> {
    compiled: Compiled<F, G>,
    error: PhantomData<E>,
}

impl<F, G> RetainedSlice for RetainedSliceImpl<F, G, ()>
where
    F: FnMut(&[Array]) -> Vec<Array>,
    G: FnMut(&[Array]) -> Vec<Array>,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        CallMut::<&[Array], Vec<Array>, ()>::call_mut(&mut self.compiled, args)
    }
}

impl<F, G> RetainedSlice for RetainedSliceImpl<F, G, Exception>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        CallMut::<&[Array], Vec<Array>, Exception>::call_mut(&mut self.compiled, args)
    }
}

impl<F, G> RetainedUnary for RetainedUnaryImpl<F, G, ()>
where
    F: FnMut(&Array) -> Array,
    G: FnMut(&[Array]) -> Vec<Array>,
{
    fn call_mut(&mut self, arg: &Array) -> Result<Array, Exception> {
        CallMut::<&Array, Array, ()>::call_mut(&mut self.compiled, arg)
    }
}

impl<F, G> RetainedUnary for RetainedUnaryImpl<F, G, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
{
    fn call_mut(&mut self, arg: &Array) -> Result<Array, Exception> {
        CallMut::<&Array, Array, Exception>::call_mut(&mut self.compiled, arg)
    }
}

impl<F, G> RetainedBinary for RetainedBinaryImpl<F, G, ()>
where
    F: FnMut((&Array, &Array)) -> Array,
    G: FnMut(&[Array]) -> Vec<Array>,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        CallMut::<(&Array, &Array), Array, ()>::call_mut(&mut self.compiled, args)
    }
}

impl<F, G> RetainedBinary for RetainedBinaryImpl<F, G, Exception>
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        CallMut::<(&Array, &Array), Array, Exception>::call_mut(&mut self.compiled, args)
    }
}

impl<F, G> RetainedTernary for RetainedTernaryImpl<F, G, ()>
where
    F: FnMut((&Array, &Array, &Array)) -> Array,
    G: FnMut(&[Array]) -> Vec<Array>,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        CallMut::<(&Array, &Array, &Array), Array, ()>::call_mut(&mut self.compiled, args)
    }
}

impl<F, G> RetainedTernary for RetainedTernaryImpl<F, G, Exception>
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        CallMut::<(&Array, &Array, &Array), Array, Exception>::call_mut(&mut self.compiled, args)
    }
}

/// A trait for functions that can be compiled.
///
/// # Generic parameters
///
/// - `A`: The type of the array arguments
/// - `O`: The type of the output
/// - `E`: The type of the error
pub trait Compile<A, O, E>: Sized {
    /// The type of the arguments that the returned closure takes.
    ///
    /// This is needed to relax the lifetime requirements of the returned
    /// closure. Otherwise, the arguments to the returned closure would have to
    /// live longer than the closure itself.
    type Args<'a>;

    /// Compiles the function.
    fn compile<'args>(self, shapeless: bool) -> impl CallMut<Self::Args<'args>, O, E>;
}

impl<F> Compile<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile<'args>(self, shapeless: bool) -> impl CallMut<Self::Args<'args>, Vec<Array>, ()> {
        let state = CompiledState {
            f: self,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    type Args<'a> = &'a Array;

    fn compile<'args>(mut self, shapeless: bool) -> impl CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)(&args[0]);
            vec![result]
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args>(mut self, shapeless: bool) -> impl CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args>(mut self, shapeless: bool) -> impl CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile<'args>(
        self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Vec<Array>, Exception> {
        let state = CompiledState {
            f: self,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = &'a Array;

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(&args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState {
            f,
            shapeless,
            lease: new_compile_lease(),
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> CompileRetained<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Callable = Box<dyn RetainedSlice>;

    fn compile_retained(self, shapeless: bool) -> Self::Callable {
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: self,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedSliceImpl {
            compiled,
            error: PhantomData::<()>,
        })
    }
}

impl<F> CompileRetained<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    type Callable = Box<dyn RetainedUnary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function = move |args: &[Array]| -> Vec<Array> { vec![(self)(&args[0])] };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedUnaryImpl {
            compiled,
            error: PhantomData::<()>,
        })
    }
}

impl<F> CompileRetained<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    type Callable = Box<dyn RetainedBinary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function = move |args: &[Array]| -> Vec<Array> { vec![(self)((&args[0], &args[1]))] };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedBinaryImpl {
            compiled,
            error: PhantomData::<()>,
        })
    }
}

impl<F> CompileRetained<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    type Callable = Box<dyn RetainedTernary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function =
            move |args: &[Array]| -> Vec<Array> { vec![(self)((&args[0], &args[1], &args[2]))] };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedTernaryImpl {
            compiled,
            error: PhantomData::<()>,
        })
    }
}

impl<F> CompileRetained<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Callable = Box<dyn RetainedSlice>;

    fn compile_retained(self, shapeless: bool) -> Self::Callable {
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: self,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedSliceImpl {
            compiled,
            error: PhantomData::<Exception>,
        })
    }
}

impl<F> CompileRetained<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    type Callable = Box<dyn RetainedUnary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function =
            move |args: &[Array]| -> Result<Vec<Array>, Exception> { Ok(vec![(self)(&args[0])?]) };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedUnaryImpl {
            compiled,
            error: PhantomData::<Exception>,
        })
    }
}

impl<F> CompileRetained<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Callable = Box<dyn RetainedBinary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            Ok(vec![(self)((&args[0], &args[1]))?])
        };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedBinaryImpl {
            compiled,
            error: PhantomData::<Exception>,
        })
    }
}

impl<F> CompileRetained<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Callable = Box<dyn RetainedTernary>;

    fn compile_retained(mut self, shapeless: bool) -> Self::Callable {
        let function = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            Ok(vec![(self)((&args[0], &args[1], &args[2]))?])
        };
        let compiled = Compiled {
            f_marker: PhantomData::<F>,
            state: CompiledState {
                f: function,
                shapeless,
                lease: new_compile_lease(),
            },
        };
        Box::new(RetainedTernaryImpl {
            compiled,
            error: PhantomData::<Exception>,
        })
    }
}

/// A trait for a compiled function that can be called.
pub trait CallMut<A, O, E> {
    /// Calls the compiled function with the given arguments.
    fn call_mut(&mut self, args: A) -> Result<O, Exception>;
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&Array) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0, args.1, args.2];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = &[args];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

#[inline]
fn call_mut_inner(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    args: &[impl AsRef<Array>],
) -> crate::error::Result<Vec<Array>> {
    // note: this will use the cached compile (via the id)
    // but will be able to re-evaluate with fresh state if needed
    let compiled = Closure::try_from_op(|res| unsafe {
        let constants = &[];
        mlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })?;

    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    let result_len = result_plus_state_output.len();
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

impl<F> CompiledState<F> {
    fn call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array>,
    {
        let inner_closure = Closure::new(&mut self.f);

        call_mut_inner(inner_closure, self.lease.id, self.shapeless, args)
    }

    fn fallible_call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
    {
        let inner_closure = Closure::new_fallible(&mut self.f);

        call_mut_inner(inner_closure, self.lease.id, self.shapeless, args)
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        Array,
    };

    use super::{compile, compile_retained, CallMut};

    #[test]
    fn compile_leases_have_unique_process_local_ids_and_erase_on_final_drop() {
        let first = super::new_compile_lease();
        let first_id = first.id;
        let sibling = first.clone();
        let second = super::new_compile_lease();

        assert_ne!(first.id, second.id);
        drop(first);
        assert_eq!(sibling.id, first_id, "a clone keeps the shared lease alive");
    }

    #[test]
    fn test_compile() {
        // This unit test is modified from the mlx-swift codebase

        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] * &inputs[1]] };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).drain(0..1).next().unwrap();
        // evaluate compiled
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1]).map(|x| vec![x])
        };

        // Success case
        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);

        // Error case
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        // The cache is keyed by compiled-handle identity and argument shapes.
        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        // evaluate directly
        let result = f(&args);
        assert!(result.is_err());

        // evaluate compiled
        let mut compiled = compile(f, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_with_one_arg() {
        let f = |x: &Array| x * x;

        let i = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f(&i);

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&i).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_two_args() {
        let f = |(x, y): (&Array, &Array)| x * y;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2));

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2)).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_three_args() {
        let f = |(x, y, z): (&Array, &Array, &Array)| x * y * z;
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let i3 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2, &i3));

        // evaluate compiled
        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_retained_supports_every_advertised_arity_and_fallibility() {
        let x = array!([1.0f32, 2.0]);
        let y = array!([3.0f32, 4.0]);
        let z = array!([5.0f32, 6.0]);

        let mut infallible_slice =
            compile_retained(|args: &[Array]| vec![&args[0] * &args[1]], true);
        let mut fallible_slice = compile_retained(
            |args: &[Array]| multiply(&args[0], &args[1]).map(|out| vec![out]),
            true,
        );
        let slice_args = [x.clone(), y.clone()];
        assert_eq!(
            infallible_slice.call_mut(&slice_args).unwrap()[0],
            fallible_slice.call_mut(&slice_args).unwrap()[0]
        );

        let mut infallible_unary = compile_retained(|arg: &Array| arg * arg, true);
        let mut fallible_unary = compile_retained(|arg: &Array| multiply(arg, arg), true);
        assert_eq!(
            infallible_unary.call_mut(&x).unwrap(),
            fallible_unary.call_mut(&x).unwrap()
        );

        let mut infallible_binary =
            compile_retained(|(lhs, rhs): (&Array, &Array)| lhs * rhs, true);
        let mut fallible_binary =
            compile_retained(|(lhs, rhs): (&Array, &Array)| multiply(lhs, rhs), true);
        assert_eq!(
            infallible_binary.call_mut((&x, &y)).unwrap(),
            fallible_binary.call_mut((&x, &y)).unwrap()
        );

        let mut infallible_ternary =
            compile_retained(|(a, b, c): (&Array, &Array, &Array)| a * b * c, true);
        let mut fallible_ternary = compile_retained(
            |(a, b, c): (&Array, &Array, &Array)| multiply(&multiply(a, b)?, c),
            true,
        );
        assert_eq!(
            infallible_ternary.call_mut((&x, &y, &z)).unwrap(),
            fallible_ternary.call_mut((&x, &y, &z)).unwrap()
        );
    }

    fn captured_slice(scale: f32) -> impl FnMut(&[Array]) -> Result<Vec<Array>, Exception> {
        move |args| {
            let product = multiply(&args[0], &args[1])?;
            Ok(vec![multiply(product, Array::from_f32(scale))?])
        }
    }

    fn captured_infallible_slice(scale: f32) -> impl FnMut(&[Array]) -> Vec<Array> {
        move |args| {
            let product = multiply(&args[0], &args[1]).unwrap();
            vec![multiply(product, Array::from_f32(scale)).unwrap()]
        }
    }

    fn captured_unary(scale: f32) -> impl FnMut(&Array) -> Result<Array, Exception> {
        move |x| multiply(x, Array::from_f32(scale))
    }

    fn captured_infallible_unary(scale: f32) -> impl FnMut(&Array) -> Array {
        move |x| multiply(x, Array::from_f32(scale)).unwrap()
    }

    fn captured_binary(scale: f32) -> impl FnMut((&Array, &Array)) -> Result<Array, Exception> {
        move |(x, y)| multiply(multiply(x, y)?, Array::from_f32(scale))
    }

    fn captured_infallible_binary(scale: f32) -> impl FnMut((&Array, &Array)) -> Array {
        move |(x, y)| multiply(multiply(x, y).unwrap(), Array::from_f32(scale)).unwrap()
    }

    fn captured_ternary(
        scale: f32,
    ) -> impl FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> {
        move |(x, y, z)| multiply(multiply(multiply(x, y)?, z)?, Array::from_f32(scale))
    }

    fn captured_infallible_ternary(scale: f32) -> impl FnMut((&Array, &Array, &Array)) -> Array {
        move |(x, y, z)| {
            multiply(
                multiply(multiply(x, y).unwrap(), z).unwrap(),
                Array::from_f32(scale),
            )
            .unwrap()
        }
    }

    #[test]
    fn same_closure_type_handles_are_independent_for_every_retained_interface() {
        let x = array!([1.0f32, 2.0]);
        let y = array!([3.0f32, 4.0]);
        let z = array!([5.0f32, 6.0]);
        let binary_args = [x.clone(), y.clone()];

        {
            let mut first = compile_retained(captured_infallible_slice(2.0), true);
            let mut second = compile_retained(captured_infallible_slice(3.0), true);
            assert_eq!(
                first.call_mut(&binary_args).unwrap()[0],
                array!([6.0f32, 16.0])
            );
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
            assert_eq!(
                first.call_mut(&binary_args).unwrap()[0],
                array!([6.0f32, 16.0])
            );
            drop(first);
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
            crate::transforms::compile::clear_cache();
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
        }
        {
            let mut first = compile_retained(captured_slice(2.0), true);
            let mut second = compile_retained(captured_slice(3.0), true);
            assert_eq!(
                first.call_mut(&binary_args).unwrap()[0],
                array!([6.0f32, 16.0])
            );
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
            assert_eq!(
                first.call_mut(&binary_args).unwrap()[0],
                array!([6.0f32, 16.0])
            );
            drop(first);
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
            crate::transforms::compile::clear_cache();
            assert_eq!(
                second.call_mut(&binary_args).unwrap()[0],
                array!([9.0f32, 24.0])
            );
        }
        {
            let mut first = compile_retained(captured_infallible_unary(2.0), true);
            let mut second = compile_retained(captured_infallible_unary(3.0), true);
            assert_eq!(first.call_mut(&x).unwrap(), array!([2.0f32, 4.0]));
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
            assert_eq!(first.call_mut(&x).unwrap(), array!([2.0f32, 4.0]));
            drop(first);
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
            crate::transforms::compile::clear_cache();
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
        }
        {
            let mut first = compile_retained(captured_unary(2.0), true);
            let mut second = compile_retained(captured_unary(3.0), true);
            assert_eq!(first.call_mut(&x).unwrap(), array!([2.0f32, 4.0]));
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
            assert_eq!(first.call_mut(&x).unwrap(), array!([2.0f32, 4.0]));
            drop(first);
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
            crate::transforms::compile::clear_cache();
            assert_eq!(second.call_mut(&x).unwrap(), array!([3.0f32, 6.0]));
        }
        {
            let mut first = compile_retained(captured_infallible_binary(2.0), true);
            let mut second = compile_retained(captured_infallible_binary(3.0), true);
            assert_eq!(first.call_mut((&x, &y)).unwrap(), array!([6.0f32, 16.0]));
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
            assert_eq!(first.call_mut((&x, &y)).unwrap(), array!([6.0f32, 16.0]));
            drop(first);
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
            crate::transforms::compile::clear_cache();
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
        }
        {
            let mut first = compile_retained(captured_binary(2.0), true);
            let mut second = compile_retained(captured_binary(3.0), true);
            assert_eq!(first.call_mut((&x, &y)).unwrap(), array!([6.0f32, 16.0]));
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
            assert_eq!(first.call_mut((&x, &y)).unwrap(), array!([6.0f32, 16.0]));
            drop(first);
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
            crate::transforms::compile::clear_cache();
            assert_eq!(second.call_mut((&x, &y)).unwrap(), array!([9.0f32, 24.0]));
        }
        {
            let mut first = compile_retained(captured_infallible_ternary(2.0), true);
            let mut second = compile_retained(captured_infallible_ternary(3.0), true);
            assert_eq!(
                first.call_mut((&x, &y, &z)).unwrap(),
                array!([30.0f32, 96.0])
            );
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
            assert_eq!(
                first.call_mut((&x, &y, &z)).unwrap(),
                array!([30.0f32, 96.0])
            );
            drop(first);
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
            crate::transforms::compile::clear_cache();
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
        }
        {
            let mut first = compile_retained(captured_ternary(2.0), true);
            let mut second = compile_retained(captured_ternary(3.0), true);
            assert_eq!(
                first.call_mut((&x, &y, &z)).unwrap(),
                array!([30.0f32, 96.0])
            );
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
            assert_eq!(
                first.call_mut((&x, &y, &z)).unwrap(),
                array!([30.0f32, 96.0])
            );
            drop(first);
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
            crate::transforms::compile::clear_cache();
            assert_eq!(
                second.call_mut((&x, &y, &z)).unwrap(),
                array!([45.0f32, 144.0])
            );
        }
    }

    fn traced_square(traces: Arc<AtomicUsize>) -> impl FnMut(&Array) -> Result<Array, Exception> {
        move |x| {
            traces.fetch_add(1, Ordering::Relaxed);
            multiply(x, x)
        }
    }

    fn captured_scale(
        scale: f32,
        traces: Arc<AtomicUsize>,
    ) -> impl FnMut(&Array) -> Result<Array, Exception> {
        move |x| {
            traces.fetch_add(1, Ordering::Relaxed);
            multiply(x, Array::from_f32(scale))
        }
    }

    #[test]
    fn same_closure_type_retained_handles_have_independent_backend_state() {
        let first_traces = Arc::new(AtomicUsize::new(0));
        let second_traces = Arc::new(AtomicUsize::new(0));
        let mut first = compile_retained(captured_scale(2.0, Arc::clone(&first_traces)), true);
        let mut second = compile_retained(captured_scale(3.0, Arc::clone(&second_traces)), true);
        let x = array!([2.0f32, 4.0]);

        // Alternate the two handles. A TypeId-only backend key aliases these same-type closures and
        // returns one capture's graph for both callables.
        assert_eq!(first.call_mut(&x).unwrap(), array!([4.0f32, 8.0]));
        assert_eq!(second.call_mut(&x).unwrap(), array!([6.0f32, 12.0]));
        assert_eq!(first.call_mut(&x).unwrap(), array!([4.0f32, 8.0]));
        assert_eq!(second.call_mut(&x).unwrap(), array!([6.0f32, 12.0]));
        assert_eq!(first_traces.load(Ordering::Relaxed), 1);
        assert_eq!(second_traces.load(Ordering::Relaxed), 1);

        // Dropping one independent lease must not erase the other's compiled graph.
        drop(first);
        assert_eq!(second.call_mut(&x).unwrap(), array!([6.0f32, 12.0]));
        assert_eq!(second_traces.load(Ordering::Relaxed), 1);

        // Explicit cache invalidation keeps the stable handle identity but forces exactly one
        // retrace, after which the handle hits again.
        crate::transforms::compile::clear_cache();
        assert_eq!(second.call_mut(&x).unwrap(), array!([6.0f32, 12.0]));
        assert_eq!(second.call_mut(&x).unwrap(), array!([6.0f32, 12.0]));
        assert_eq!(second_traces.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn cloned_compiled_state_erases_only_after_the_final_owner_drops() {
        let traces = Arc::new(AtomicUsize::new(0));
        let trace = Arc::clone(&traces);
        let function = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            trace.fetch_add(1, Ordering::Relaxed);
            Ok(vec![multiply(&args[0], &args[0])?])
        };
        let mut first = super::Compiled {
            f_marker: PhantomData::<fn(&[Array]) -> Result<Vec<Array>, Exception>>,
            state: super::CompiledState {
                f: function,
                shapeless: true,
                lease: super::new_compile_lease(),
            },
        };
        let mut sibling = first.clone();
        let x = array!([2.0f32, 3.0]);

        assert_eq!(
            CallMut::<&[Array], Vec<Array>, Exception>::call_mut(
                &mut first,
                std::slice::from_ref(&x),
            )
            .unwrap()[0],
            array!([4.0f32, 9.0])
        );
        drop(first);
        assert_eq!(
            CallMut::<&[Array], Vec<Array>, Exception>::call_mut(
                &mut sibling,
                std::slice::from_ref(&x),
            )
            .unwrap()[0],
            array!([4.0f32, 9.0])
        );
        assert_eq!(
            traces.load(Ordering::Relaxed),
            1,
            "dropping one clone must not erase the shared backend graph"
        );
    }

    #[test]
    fn test_compile_retained_traces_once_across_distinct_argument_lifetimes() {
        let traces = Arc::new(AtomicUsize::new(0));
        let mut compiled = compile_retained(traced_square(Arc::clone(&traces)), true);

        {
            let x = array!([2.0f32, 3.0]);
            assert_eq!(compiled.call_mut(&x).unwrap(), array!([4.0f32, 9.0]));
        }
        {
            // A different lexical lifetime and shape exercise the object-safe, lifetime-generic
            // call surface and the shape-polymorphic retained handle.
            let x = array!([2.0f32, 3.0, 4.0]);
            assert_eq!(compiled.call_mut(&x).unwrap(), array!([4.0f32, 9.0, 16.0]));
        }

        assert_eq!(
            traces.load(Ordering::Relaxed),
            1,
            "the retained handle must reuse the traced function"
        );

        drop(compiled);
        let mut replacement = compile_retained(traced_square(Arc::clone(&traces)), true);
        let x = array!([5.0f32]);
        assert_eq!(replacement.call_mut(&x).unwrap(), array!([25.0f32]));
        assert_eq!(
            traces.load(Ordering::Relaxed),
            2,
            "dropping the retained handle must erase its backend compiled state"
        );
    }
}
