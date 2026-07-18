use mlx_rs::{argmax_axis, array, categorical, error::Exception, random::RandomState, Array};

pub trait Sampler {
    fn sample(&mut self, logits: &Array, temp: f32) -> Result<Array, Exception>;
}

/// Temperature sampler backed by its own random state.
///
/// An explicit key is supplied for every stochastic draw. This avoids the
/// synchronous global-RNG state update required by MLX 0.32 while retaining
/// lazy execution. Use [`Self::with_seed`] for reproducible sampling.
/// `mlx_rs::random::seed` does not influence sampler-owned state.
pub struct DefaultSampler {
    state: Option<RandomState>,
}

/// Backward-compatible value constructor for the former unit struct.
#[allow(non_upper_case_globals)]
pub const DefaultSampler: DefaultSampler = DefaultSampler { state: None };

impl DefaultSampler {
    pub fn new() -> Self {
        Self { state: None }
    }

    pub fn with_seed(seed: u64) -> Result<Self, Exception> {
        Ok(Self {
            state: Some(RandomState::with_seed(seed)?),
        })
    }

    /// Reset this sampler to a reproducible seed.
    pub fn seed(&mut self, seed: u64) -> Result<(), Exception> {
        self.state = Some(RandomState::with_seed(seed)?);
        Ok(())
    }
}

impl Default for DefaultSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for DefaultSampler {
    fn sample(&mut self, logits: &Array, temp: f32) -> Result<Array, Exception> {
        match temp {
            0.0 => argmax_axis!(logits, -1),
            _ => {
                let logits = logits.multiply(array!(1.0 / temp))?;
                if self.state.is_none() {
                    self.state = Some(RandomState::new()?);
                }
                let key = self.state.as_mut().unwrap().next_key()?;
                categorical!(logits, key = &key)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random::{key, seed, uniform};

    fn logits() -> Array {
        uniform::<_, f32>(0.0, 8.0, &[1, 64], &key(11).unwrap()).unwrap()
    }

    #[test]
    fn seeded_sampler_is_reproducible_and_seed_sensitive() {
        let logits = logits();
        let draw = |mut sampler: DefaultSampler| -> Vec<u32> {
            (0..16)
                .map(|_| sampler.sample(&logits, 0.7).unwrap().item::<u32>())
                .collect()
        };
        let a = draw(DefaultSampler::with_seed(42).unwrap());
        assert_eq!(a, draw(DefaultSampler::with_seed(42).unwrap()));
        assert_ne!(a, draw(DefaultSampler::with_seed(43).unwrap()));
    }

    #[test]
    fn sampler_does_not_consume_global_rng() {
        let logits = logits();
        seed(1234).unwrap();
        let expected = uniform::<_, f32>(0.0, 1.0, None, None)
            .unwrap()
            .item::<f32>();
        seed(1234).unwrap();
        let mut sampler = DefaultSampler::with_seed(0).unwrap();
        for _ in 0..8 {
            sampler.sample(&logits, 0.7).unwrap().item::<u32>();
        }
        let actual = uniform::<_, f32>(0.0, 1.0, None, None)
            .unwrap()
            .item::<f32>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn greedy_sampling_is_seed_independent() {
        let logits = logits();
        let a = DefaultSampler::with_seed(1)
            .unwrap()
            .sample(&logits, 0.0)
            .unwrap()
            .item::<u32>();
        let b = DefaultSampler::with_seed(2)
            .unwrap()
            .sample(&logits, 0.0)
            .unwrap()
            .item::<u32>();
        assert_eq!(a, b);
    }

    /// Decode-shaped timing for the global fallback versus sampler-owned RNG.
    ///
    /// Run on Apple Silicon with:
    /// `cargo test -p mlx-lm --release decode_rng_paths -- --ignored --nocapture`
    #[test]
    #[ignore = "manual performance measurement"]
    fn decode_rng_paths() {
        use mlx_rs::random::categorical;
        use std::time::Instant;

        const VOCAB: i32 = 151_936;
        const WARMUP: usize = 20;
        const ITERS: usize = 200;

        let logits = uniform::<_, f32>(0.0, 1.0, &[1, VOCAB], &key(7).unwrap()).unwrap();
        logits.eval().unwrap();

        let global_step = || {
            let scaled = logits.multiply(array!(1.0 / 0.7)).unwrap();
            categorical(&scaled, None, None, Option::<&Array>::None)
                .unwrap()
                .item::<u32>()
        };

        for _ in 0..WARMUP {
            global_step();
        }
        let global_start = Instant::now();
        for _ in 0..ITERS {
            global_step();
        }
        let global = global_start.elapsed();

        let mut sampler = DefaultSampler::with_seed(42).unwrap();
        for _ in 0..WARMUP {
            sampler.sample(&logits, 0.7).unwrap().item::<u32>();
        }
        let owned_start = Instant::now();
        for _ in 0..ITERS {
            sampler.sample(&logits, 0.7).unwrap().item::<u32>();
        }
        let owned = owned_start.elapsed();

        println!("decode-shaped step ({ITERS} iterations, vocab {VOCAB}):");
        println!(
            "  global key path: {:9.1} us/iter",
            global.as_secs_f64() * 1e6 / ITERS as f64
        );
        println!(
            "  owned RNG state: {:9.1} us/iter",
            owned.as_secs_f64() * 1e6 / ITERS as f64
        );
    }
}
