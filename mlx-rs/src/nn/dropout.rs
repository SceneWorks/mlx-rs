use crate::module::Module;
use crate::Array;
use crate::{
    array,
    error::Exception,
    ops::multiply,
    random::{bernoulli, RandomState},
};
use mlx_internal_macros::{Buildable, Builder};
use mlx_macros::ModuleParameters;

use crate::error::DropoutBuildError;

/// Builder for [`Dropout`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct DropoutBuilder {
    /// The probability of zeroing an element.
    #[builder(optional, default = Dropout::DEFAULT_P)]
    p: f32,
}

fn build_dropout(builder: DropoutBuilder) -> Result<Dropout, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout {
        one_minus_p: 1.0 - p,
        training: Dropout::DEFAULT_TRAINING,
        random_state: RandomState::default(),
    })
}

/// Randomly zero a portion of the elements during training.
///
/// The remaining elements are multiplied with `1 / (1-p)` where
/// `p` is the probability of zeroing an element. This is done so the
/// expected value of a given element will remain the same.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout {
    /// `1-p`, where `p` is the probability of zeroing an element. `p` is default to
    /// [`Dropout::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,

    #[param]
    random_state: RandomState,
}

impl Dropout {
    /// Default value for the probability of zeroing an element.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Set the layer-owned random state to a reproducible seed.
    pub fn with_seed(mut self, seed: u64) -> Result<Self, Exception> {
        self.random_state.seed(seed)?;
        Ok(self)
    }
}

impl Module<&Array> for Dropout {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        let p1 = array!(self.one_minus_p);
        let key = self.random_state.next_key()?;
        let mask = bernoulli(&p1, x.shape(), &key)?;
        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Builder for [`Dropout2d`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout2d,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct Dropout2dBuilder {
    /// The probability of zeroing a channel.
    #[builder(optional, default = Dropout2d::DEFAULT_P)]
    p: f32,
}

fn build_dropout2d(builder: Dropout2dBuilder) -> Result<Dropout2d, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout2d {
        one_minus_p: 1.0 - p,
        training: Dropout2d::DEFAULT_TRAINING,
        random_state: RandomState::default(),
    })
}

/// Apply 2D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e. the input shape should be
/// `NWHC` or `WHC` where:`N` is the batch dimension,`H` is the input
/// image height,`W` is the input image width, and`C` is the number of
/// input channels
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// beneficial for early convolution layers where adjacent pixels are
/// correlated. In such case, traditional dropout may not effectively
/// regularize activations. For more details, see [1].
///
/// [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
/// Efficient Object Localization Using Convolutional Networks. CVPR 2015.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout2d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout2d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout2d::DEFAULT_TRAINING`] if not
    /// specified. Default to [`Dropout2d::DEFAULT_TRAINING`] if not specified.
    pub training: bool,

    #[param]
    random_state: RandomState,
}

impl Dropout2d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Set the layer-owned random state to a reproducible seed.
    pub fn with_seed(mut self, seed: u64) -> Result<Self, Exception> {
        self.random_state.seed(seed)?;
        Ok(self)
    }
}

impl Module<&Array> for Dropout2d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 3 && ndim != 4 {
            return Err(Exception::custom("Expecting 3D or 4D input"));
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 3D input: (1, 1, C)
        // 4D input: (B, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len - 2] = 1;
        mask_shape[len - 3] = 1;

        let p1 = array!(self.one_minus_p);
        let key = self.random_state.next_key()?;
        let mask = bernoulli(&p1, &mask_shape, &key)?;

        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// Builder for [`Dropout3d`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_dropout3d,
    default_infallible,
    err = DropoutBuildError,
)]
pub struct Dropout3dBuilder {
    /// The probability of zeroing a channel.
    #[builder(optional, default = Dropout3d::DEFAULT_P)]
    p: f32,
}

fn build_dropout3d(builder: Dropout3dBuilder) -> Result<Dropout3d, DropoutBuildError> {
    let p = builder.p;

    if !(0.0..1.0).contains(&p) {
        return Err(DropoutBuildError::InvalidProbability);
    }

    Ok(Dropout3d {
        one_minus_p: 1.0 - p,
        training: Dropout3d::DEFAULT_TRAINING,
        random_state: RandomState::default(),
    })
}

/// Apply 3D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e., the input shape should be
/// `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
/// `H` is the input image height, `W` is the input image width, and `C` is
/// the number of input channels.
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// often beneficial for convolutional layers processing 3D data, like in
/// medical imaging or video processing.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Dropout3d {
    /// `1-p`, where `p` is the probability of zeroing a channel. `p` is default to
    /// [`Dropout3d::DEFAULT_P`] if not specified.
    pub one_minus_p: f32,

    /// Whether the layer is in training mode. Default to [`Dropout3d::DEFAULT_TRAINING`] if not
    /// specified.
    pub training: bool,

    #[param]
    random_state: RandomState,
}

impl Dropout3d {
    /// Default value for the probability of zeroing a channel.
    pub const DEFAULT_P: f32 = 0.5;

    /// Default value for the training mode.
    pub const DEFAULT_TRAINING: bool = true;

    /// Set the layer-owned random state to a reproducible seed.
    pub fn with_seed(mut self, seed: u64) -> Result<Self, Exception> {
        self.random_state.seed(seed)?;
        Ok(self)
    }
}

impl Module<&Array> for Dropout3d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let ndim = x.ndim();

        if ndim != 4 && ndim != 5 {
            return Err(Exception::custom("Expecting 4D or 5D input"));
        }

        if self.one_minus_p == 1.0 || !self.training {
            return Ok(x.clone());
        }

        // Dropout is applied on the whole channel
        // 4D input: (1, 1, 1, C)
        // 5D input: (B, 1, 1, 1, C)

        let mut mask_shape = x.shape().to_vec();
        let len = mask_shape.len();
        mask_shape[len - 2] = 1;
        mask_shape[len - 3] = 1;
        mask_shape[len - 4] = 1;

        let p1 = array!(self.one_minus_p);
        let key = self.random_state.next_key()?;
        let mask = bernoulli(&p1, &mask_shape, &key)?;

        multiply(multiply(array!(1.0 / self.one_minus_p), mask)?, x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.training = mode;
    }
}

// The following tests were ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::module::ModuleParameters;
    use crate::random::{key, uniform};
    use crate::transforms::compile::compile_with_state;
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_dropout() {
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key(959).unwrap()).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(a.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.06);
        assert_float_eq!(a.sum(None).unwrap().item::<f32>(), 128.0, abs <= 15.36);
        let result = Dropout::new().with_seed(959).unwrap().forward(&a).unwrap();
        let repeated = Dropout::new().with_seed(959).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(result.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.1);
        assert_float_eq!(result.sum(None).unwrap().item::<f32>(), 128.0, abs <= 25.6);
        assert_eq!(result.as_slice::<f32>(), repeated.as_slice::<f32>());
    }

    #[test]
    fn test_dropout2d() {
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key(695).unwrap()).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(a.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.06);
        assert_float_eq!(a.sum(None).unwrap().item::<f32>(), 128.0, abs <= 15.36);
        let result = Dropout2d::new()
            .with_seed(695)
            .unwrap()
            .forward(&a)
            .unwrap();
        let repeated = Dropout2d::new()
            .with_seed(695)
            .unwrap()
            .forward(&a)
            .unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(result.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.25);
        assert_float_eq!(result.sum(None).unwrap().item::<f32>(), 128.0, abs <= 64.0);
        assert_eq!(result.as_slice::<f32>(), repeated.as_slice::<f32>());
    }

    #[test]
    fn test_dropout3d() {
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 8, 4], &key(23).unwrap()).unwrap();
        assert_eq!(a.shape(), &[2, 8, 8, 4]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);
        assert_float_eq!(a.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.06);
        assert_float_eq!(a.sum(None).unwrap().item::<f32>(), 256.0, abs <= 30.72);
        let result = Dropout3d::new().with_seed(23).unwrap().forward(&a).unwrap();
        let repeated = Dropout3d::new().with_seed(23).unwrap().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8, 4]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(result.mean(None).unwrap().item::<f32>(), 0.5, abs <= 0.3);
        assert_float_eq!(result.sum(None).unwrap().item::<f32>(), 256.0, abs <= 153.6);
        assert_eq!(result.as_slice::<f32>(), repeated.as_slice::<f32>());
    }

    #[test]
    fn dropout_layers_do_not_consume_global_rng() {
        fn next_global() -> f32 {
            uniform::<_, f32>(0.0, 1.0, None, None)
                .unwrap()
                .item::<f32>()
        }

        crate::random::seed(1234).unwrap();
        let expected = next_global();

        crate::random::seed(1234).unwrap();
        Dropout::new()
            .with_seed(1)
            .unwrap()
            .forward(&Array::ones::<f32>(&[2, 8, 16]).unwrap())
            .unwrap()
            .eval()
            .unwrap();
        assert_eq!(expected, next_global());

        crate::random::seed(1234).unwrap();
        Dropout2d::new()
            .with_seed(2)
            .unwrap()
            .forward(&Array::ones::<f32>(&[2, 8, 16]).unwrap())
            .unwrap()
            .eval()
            .unwrap();
        assert_eq!(expected, next_global());

        crate::random::seed(1234).unwrap();
        Dropout3d::new()
            .with_seed(3)
            .unwrap()
            .forward(&Array::ones::<f32>(&[2, 8, 8, 4]).unwrap())
            .unwrap()
            .eval()
            .unwrap();
        assert_eq!(expected, next_global());
    }

    #[test]
    fn compiled_dropout_advances_and_replays_seeded_state() {
        let x = Array::ones::<f32>(&[256]).unwrap();
        let run = |seed| {
            let mut layer = Dropout::new().with_seed(seed).unwrap();
            let mut compiled =
                compile_with_state(|layer: &mut Dropout, x: &Array| layer.forward(x), None);
            let first = compiled(&mut layer, &x).unwrap();
            let second = compiled(&mut layer, &x).unwrap();
            first.eval().unwrap();
            second.eval().unwrap();
            (
                first.as_slice::<f32>().to_vec(),
                second.as_slice::<f32>().to_vec(),
            )
        };

        let (first, second) = run(42);
        assert_ne!(first, second, "compiled dropout must advance its RNG state");
        assert_eq!(
            (first, second),
            run(42),
            "the same seed must replay both masks"
        );
    }

    #[test]
    fn dropout_rng_state_is_frozen_and_non_trainable() {
        let layer = Dropout::new().with_seed(42).unwrap();
        assert_eq!(layer.num_parameters(), 1);
        assert_eq!(layer.parameters().flatten().len(), 1);
        assert!(layer.trainable_parameters().flatten().is_empty());
        assert_eq!(layer.all_frozen(), Some(true));
    }
}
