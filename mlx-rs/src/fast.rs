//! Fast implementations of commonly used multi-op functions.

use std::ffi::{CStr, CString};

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, VectorArray};
use crate::{Array, Dtype, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// A template parameter for a JIT-compiled [`MetalKernel`].
///
/// Template parameters are substituted into the generated kernel at JIT-compile
/// time. They mirror MLX's `mx.fast.metal_kernel` `template` argument and the
/// underlying `mlx_fast_metal_kernel_config_add_template_arg_*` C entry points.
#[derive(Debug, Clone)]
pub enum TemplateArg {
    /// A `dtype` template parameter (e.g. `T = float16`).
    Dtype(Dtype),
    /// An integer template parameter.
    Int(i32),
    /// A boolean template parameter.
    Bool(bool),
}

impl From<Dtype> for TemplateArg {
    fn from(value: Dtype) -> Self {
        TemplateArg::Dtype(value)
    }
}

impl From<i32> for TemplateArg {
    fn from(value: i32) -> Self {
        TemplateArg::Int(value)
    }
}

impl From<bool> for TemplateArg {
    fn from(value: bool) -> Self {
        TemplateArg::Bool(value)
    }
}

/// Describes a single output of a [`MetalKernel`] dispatch: its shape and dtype.
#[derive(Debug, Clone)]
pub struct OutputArg {
    /// The shape of the output array.
    pub shape: Vec<i32>,
    /// The dtype of the output array.
    pub dtype: Dtype,
}

/// A safe, JIT-compiled custom Metal kernel built from hand-written MSL source.
///
/// This is the Rust binding for MLX's `mx.fast.metal_kernel`. The kernel source
/// is the *body* of a Metal compute function (MLX synthesizes the function
/// signature from the declared `input_names`/`output_names`). The kernel is
/// JIT-compiled and cached by MLX on first dispatch, keyed by the kernel name,
/// source, and any template parameters.
///
/// # Example
///
/// ```no_run
/// use mlx_rs::{Array, Dtype};
/// use mlx_rs::fast::{MetalKernel, OutputArg};
///
/// let kernel = MetalKernel::new(
///     "scale",
///     &["a"],
///     &["out"],
///     r#"
///         uint elem = thread_position_in_grid.x;
///         out[elem] = a[elem] * 2.0f;
///     "#,
/// )
/// .unwrap();
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
/// let outputs = kernel
///     .apply()
///     .input(&a)
///     .output(OutputArg { shape: vec![4], dtype: Dtype::Float32 })
///     .grid(4, 1, 1)
///     .thread_group(4, 1, 1)
///     .run()
///     .unwrap();
/// let out = &outputs[0];
/// # let _ = out;
/// ```
pub struct MetalKernel {
    inner: mlx_sys::mlx_fast_metal_kernel,
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.inner) };
    }
}

/// Build an `mlx_vector_string` from a slice of names.
///
/// Returns the owned C handle; the caller is responsible for freeing it via
/// `mlx_vector_string_free`.
fn new_vector_string(names: &[impl AsRef<str>]) -> Result<mlx_sys::mlx_vector_string> {
    // Hold the `CString`s alive until after `*_new_data` copies them.
    let c_strings = names
        .iter()
        .map(|n| CString::new(n.as_ref()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Exception::from(e.to_string().as_str()))?;
    let mut ptrs: Vec<*const std::os::raw::c_char> = c_strings.iter().map(|c| c.as_ptr()).collect();
    let vec = unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_mut_ptr(), ptrs.len()) };
    Ok(vec)
}

impl MetalKernel {
    /// Create (but do not yet compile) a custom Metal kernel.
    ///
    /// # Params
    ///
    /// - `name`: A name for the kernel. Combined with the source and template
    ///   parameters this forms MLX's JIT cache key.
    /// - `input_names`: The names of the input arrays, in dispatch order. These
    ///   become the input parameters of the synthesized Metal function.
    /// - `output_names`: The names of the output arrays, in dispatch order.
    /// - `source`: The MSL source for the *body* of the compute function.
    ///
    /// `ensure_row_contiguous` defaults to `true` and `atomic_outputs` to
    /// `false`, matching MLX's `mx.fast.metal_kernel` defaults. Use
    /// [`MetalKernel::with_options`] to override them.
    pub fn new(
        name: &str,
        input_names: &[impl AsRef<str>],
        output_names: &[impl AsRef<str>],
        source: &str,
    ) -> Result<Self> {
        Self::with_options(name, input_names, output_names, source, "", true, false)
    }

    /// Create a custom Metal kernel with full control over the optional flags.
    ///
    /// - `header`: Source prepended verbatim before the kernel (e.g. helper
    ///   functions or `#include`s). Pass `""` for none.
    /// - `ensure_row_contiguous`: If `true`, MLX ensures inputs are row
    ///   contiguous before the kernel runs.
    /// - `atomic_outputs`: If `true`, output buffers are declared atomic.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        name: &str,
        input_names: &[impl AsRef<str>],
        output_names: &[impl AsRef<str>],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        let c_name = CString::new(name).map_err(|e| Exception::from(e.to_string().as_str()))?;
        let c_source = CString::new(source).map_err(|e| Exception::from(e.to_string().as_str()))?;
        let c_header = CString::new(header).map_err(|e| Exception::from(e.to_string().as_str()))?;

        let inputs = new_vector_string(input_names)?;
        let outputs = new_vector_string(output_names)?;

        let inner = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                inputs,
                outputs,
                c_source.as_ptr(),
                c_header.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };

        // `*_new` copies the vectors; free our owned handles.
        unsafe {
            mlx_sys::mlx_vector_string_free(inputs);
            mlx_sys::mlx_vector_string_free(outputs);
        }

        if inner.ctx.is_null() {
            // The C layer routes the C++ exception through the global handler.
            let what = crate::error::get_and_clear_last_mlx_error()
                .map(|e| e.what)
                .unwrap_or_else(|| "failed to create metal kernel".to_string());
            return Err(Exception::from(what.as_str()));
        }

        Ok(Self { inner })
    }

    /// Begin building a dispatch of this kernel. See [`MetalKernelDispatch`].
    pub fn apply(&self) -> MetalKernelDispatch<'_> {
        MetalKernelDispatch {
            kernel: self,
            inputs: Vec::new(),
            outputs: Vec::new(),
            grid: (1, 1, 1),
            thread_group: (1, 1, 1),
            template_args: Vec::new(),
            init_value: None,
            verbose: false,
        }
    }
}

/// A builder for a single dispatch of a [`MetalKernel`].
///
/// Created via [`MetalKernel::apply`]. Configure the inputs, outputs, grid and
/// threadgroup dimensions (and optionally template params / init value), then
/// call [`MetalKernelDispatch::run`] to JIT-compile (on first use) and execute,
/// returning the output [`Array`]s in declared order.
#[derive(Debug)]
pub struct MetalKernelDispatch<'a> {
    kernel: &'a MetalKernel,
    inputs: Vec<&'a Array>,
    outputs: Vec<OutputArg>,
    grid: (i32, i32, i32),
    thread_group: (i32, i32, i32),
    template_args: Vec<(String, TemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
}

impl<'a> MetalKernelDispatch<'a> {
    /// Append an input array (in the order the kernel's `input_names` declared).
    pub fn input(mut self, array: &'a Array) -> Self {
        self.inputs.push(array);
        self
    }

    /// Append all input arrays from an iterator.
    pub fn inputs(mut self, arrays: impl IntoIterator<Item = &'a Array>) -> Self {
        self.inputs.extend(arrays);
        self
    }

    /// Declare an output (shape + dtype), in the order `output_names` declared.
    pub fn output(mut self, output: OutputArg) -> Self {
        self.outputs.push(output);
        self
    }

    /// Declare an output by shape and dtype.
    pub fn output_shape(mut self, shape: impl Into<Vec<i32>>, dtype: Dtype) -> Self {
        self.outputs.push(OutputArg {
            shape: shape.into(),
            dtype,
        });
        self
    }

    /// Set the dispatch grid dimensions (total threads per dimension).
    pub fn grid(mut self, x: i32, y: i32, z: i32) -> Self {
        self.grid = (x, y, z);
        self
    }

    /// Set the threadgroup dimensions.
    pub fn thread_group(mut self, x: i32, y: i32, z: i32) -> Self {
        self.thread_group = (x, y, z);
        self
    }

    /// Add a template parameter (substituted at JIT-compile time).
    pub fn template_arg(mut self, name: impl Into<String>, value: impl Into<TemplateArg>) -> Self {
        self.template_args.push((name.into(), value.into()));
        self
    }

    /// Set the initial value used to fill outputs before the kernel runs.
    ///
    /// Required when the kernel only writes some output elements (e.g. atomic
    /// accumulation). Mirrors MLX's `init_value`.
    pub fn init_value(mut self, value: f32) -> Self {
        self.init_value = Some(value);
        self
    }

    /// Enable MLX's verbose mode (prints the generated kernel source).
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// JIT-compile (on first use) and dispatch the kernel on the default stream.
    ///
    /// Returns the output [`Array`]s in the order their `output_names` were
    /// declared. Errors are surfaced from MLX (e.g. MSL compile failures, or
    /// running without the `metal` feature / on a non-Metal device).
    pub fn run(self) -> Result<Vec<Array>> {
        self.run_device(crate::StreamOrDevice::default())
    }

    /// JIT-compile (on first use) and dispatch the kernel on the given stream.
    ///
    /// Returns the output [`Array`]s in the order their `output_names` were
    /// declared.
    pub fn run_device(self, stream: impl AsRef<Stream>) -> Result<Vec<Array>> {
        let config = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };

        // Build the config; on any error, free it before returning.
        let build = || -> Result<()> {
            for out in &self.outputs {
                let status = unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config,
                        out.shape.as_ptr(),
                        out.shape.len(),
                        out.dtype.into(),
                    )
                };
                check_status(status)?;
            }

            check_status(unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(
                    config,
                    self.grid.0,
                    self.grid.1,
                    self.grid.2,
                )
            })?;
            check_status(unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    config,
                    self.thread_group.0,
                    self.thread_group.1,
                    self.thread_group.2,
                )
            })?;

            if let Some(value) = self.init_value {
                check_status(unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_set_init_value(config, value)
                })?;
            }

            check_status(unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_verbose(config, self.verbose)
            })?;

            for (name, value) in &self.template_args {
                let c_name = CString::new(name.as_str())
                    .map_err(|e| Exception::from(e.to_string().as_str()))?;
                let status = match value {
                    TemplateArg::Dtype(dtype) => unsafe {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config,
                            c_name.as_ptr(),
                            (*dtype).into(),
                        )
                    },
                    TemplateArg::Int(v) => unsafe {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config,
                            c_name.as_ptr(),
                            *v,
                        )
                    },
                    TemplateArg::Bool(v) => unsafe {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config,
                            c_name.as_ptr(),
                            *v,
                        )
                    },
                };
                check_status(status)?;
            }
            Ok(())
        };

        let result = (|| -> Result<Vec<Array>> {
            build()?;

            let inputs = VectorArray::try_from_iter(self.inputs.iter().copied())?;

            let outputs = VectorArray::try_from_op(|res| unsafe {
                mlx_sys::mlx_fast_metal_kernel_apply(
                    res,
                    self.kernel.inner,
                    inputs.as_ptr(),
                    config,
                    stream.as_ref().as_ptr(),
                )
            })?;

            outputs.try_into_values::<Vec<Array>>()
        })();

        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }

        result
    }
}

fn check_status(status: i32) -> Result<()> {
    if status == crate::utils::SUCCESS {
        Ok(())
    } else {
        let what = crate::error::get_and_clear_last_mlx_error()
            .map(|e| e.what)
            .unwrap_or_else(|| "metal kernel config error".to_string());
        Err(Exception::from(what.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    // Metal custom-kernel JIT round-trip tests. These dispatch a hand-written MSL
    // kernel through the new `MetalKernel` binding (sc-8529) and compare against a
    // pure-MLX-ops reference. They require the `metal` feature and an Apple Silicon
    // GPU, so they are gated on the `metal` feature.

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_elementwise_scale() {
        // Trivial round-trip: out = a * 2.0, compared to the pure-ops reference a * 2.0.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);

        let kernel = MetalKernel::new(
            "scale_by_two",
            &["a"],
            &["out"],
            r#"
                uint elem = thread_position_in_grid.x;
                out[elem] = a[elem] * 2.0f;
            "#,
        )
        .unwrap();

        let outputs = kernel
            .apply()
            .input(&a)
            .output(OutputArg {
                shape: vec![6],
                dtype: Dtype::Float32,
            })
            .grid(6, 1, 1)
            .thread_group(6, 1, 1)
            .run()
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let out = &outputs[0];
        assert_eq!(out.shape(), [6]);
        assert_eq!(out.dtype(), Dtype::Float32);

        let reference = &a * 2.0f32;
        let max_diff = (out - &reference)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max_diff < 1e-5, "max diff was {max_diff}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_two_inputs_with_template() {
        // Second case: two inputs + a dtype template param. out = a + b, compared
        // to the pure-ops reference a + b. The template `T` is substituted into the
        // generated kernel signature at JIT-compile time.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let b = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4]);

        let kernel = MetalKernel::new(
            "add_two",
            &["a", "b"],
            &["out"],
            r#"
                uint elem = thread_position_in_grid.x;
                out[elem] = a[elem] + b[elem];
            "#,
        )
        .unwrap();

        let outputs = kernel
            .apply()
            .input(&a)
            .input(&b)
            .output_shape([4], Dtype::Float32)
            .grid(4, 1, 1)
            .thread_group(4, 1, 1)
            .template_arg("T", Dtype::Float32)
            .run()
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let out = &outputs[0];
        assert_eq!(out.shape(), [4]);

        let reference = &a + &b;
        let max_diff = (out - &reference)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max_diff < 1e-5, "max diff was {max_diff}");
    }
}
