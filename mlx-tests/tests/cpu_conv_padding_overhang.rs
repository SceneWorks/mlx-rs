//! The CPU direct convolution must not write past its output when the padding is taller (or
//! wider) than the output.
//!
//! `slow_conv_2D` / `slow_conv_3D` (the CPU path for grouped / depthwise / dilated convolutions)
//! split each spatial axis into "might be out of bounds" borders and an interior. Upstream MLX
//! v0.32.0 computed the first border as `ceil(padding_lo / stride)` without clamping it to the
//! output size, so with a kernel larger than the input — a ConvNeXt 7x7 depthwise conv (padding 3)
//! on a 2x2 map, or the input-gradient conv every depthwise VJP runs (`padding_lo = k − 1 − p`) —
//! the border loop ran rows `oh >= oH` and wrote whole rows past the output buffer. The values
//! returned were right; the heap after them was not: other buffers of the same eval were silently
//! overwritten, so gradients changed from run to run (and a wide enough row segfaulted).
//!
//! `mlx-sys/patches/cpu-conv-border-clamp.patch` is upstream's fix (ml-explore/mlx af9a5924,
//! #4668): clamp every border to `[0, o]`.

use mlx_rs::ops::conv2d_device;
use mlx_rs::transforms::keyed_value_and_grad;
use mlx_rs::{Array, StreamOrDevice};
use std::collections::HashMap;
use std::rc::Rc;

fn vals(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// Host NHWC grouped convolution, stride 1, symmetric zero padding, weight `[O, kH, kW, C/g]`.
#[allow(clippy::too_many_arguments)]
fn host_conv(
    x: &[f32],
    (n, h, w, c): (usize, usize, usize, usize),
    wt: &[f32],
    (o, kh, kw): (usize, usize, usize),
    (ph, pw): (usize, usize),
    groups: usize,
    flip: bool,
) -> (Vec<f32>, (usize, usize)) {
    let (oh, ow) = (h + 2 * ph + 1 - kh, w + 2 * pw + 1 - kw);
    let (cg, og) = (c / groups, o / groups);
    let mut out = vec![0f32; n * oh * ow * o];
    for b in 0..n {
        for y in 0..oh {
            for xo in 0..ow {
                for oc in 0..o {
                    let g = oc / og;
                    let mut acc = 0f32;
                    for i in 0..kh {
                        for j in 0..kw {
                            let (wi, wj) = if flip {
                                (kh - 1 - i, kw - 1 - j)
                            } else {
                                (i, j)
                            };
                            let (iy, ix) = (y + i, xo + j);
                            if iy < ph || ix < pw || iy - ph >= h || ix - pw >= w {
                                continue;
                            }
                            let (iy, ix) = (iy - ph, ix - pw);
                            for cc in 0..cg {
                                acc += x[((b * h + iy) * w + ix) * c + g * cg + cc]
                                    * wt[((oc * kh + wi) * kw + wj) * cg + cc];
                            }
                        }
                    }
                    out[((b * oh + y) * ow + xo) * o + oc] = acc;
                }
            }
        }
    }
    (out, (oh, ow))
}

fn close(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// Upstream's regression case: a grouped conv whose height padding exceeds the one-row output.
/// The 500000-wide input makes the overhanging row 4 MB, past the end of the output allocation.
#[test]
fn grouped_conv_padding_taller_than_the_output_stays_in_the_output() {
    let cpu = StreamOrDevice::cpu();
    let xv = vals(8 * 2, 1);
    let wv = vals(2 * 5, 2);
    let x = Array::from_slice(&xv, &[1, 1, 8, 2]);
    let w = Array::from_slice(&wv, &[2, 5, 1, 1]);
    let y = conv2d_device(&x, &w, (1, 1), (2, 0), None, 2, &cpu).unwrap();
    assert_eq!(y.shape(), &[1, 1, 8, 2]);
    let (want, _) = host_conv(&xv, (1, 1, 8, 2), &wv, (2, 5, 1), (2, 0), 2, false);
    assert!(close(y.as_slice::<f32>(), &want) < 1e-5);

    let wide = Array::from_slice(&vals(500_000 * 2, 3), &[1, 1, 500_000, 2]);
    let wide_y = conv2d_device(&wide, &w, (1, 1), (2, 0), None, 2, &cpu).unwrap();
    mlx_rs::transforms::eval([&wide_y]).unwrap();
    assert_eq!(wide_y.shape(), &[1, 1, 500_000, 2]);
}

/// A ConvNeXt depthwise 7x7 (padding 3) on a 2x2 map, forward and input gradient, against the host
/// reference — repeated with the allocator's free list dirtied between runs, so a write past the
/// output lands in a buffer the next run reads.
#[test]
fn depthwise_conv_larger_than_its_input_is_repeatable_forward_and_backward() {
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let (n, h, w, c, k, p) = (2usize, 2usize, 2usize, 32usize, 7usize, 3usize);
    let xv = vals(n * h * w * c, 4);
    let wv = vals(c * k * k, 5);
    let wt = Array::from_slice(&wv, &[c as i32, k as i32, k as i32, 1]);
    let (y_ref, (oh, ow)) = host_conv(&xv, (n, h, w, c), &wv, (c, k, k), (p, p), c, false);
    assert_eq!((oh, ow), (2, 2));
    // d sum(y²)/dx = conv(2y, flipped w): the transposed depthwise conv with padding k − 1 − p.
    let dy: Vec<f32> = y_ref.iter().map(|v| 2.0 * v).collect();
    let (dx_ref, _) = host_conv(
        &dy,
        (n, oh, ow, c),
        &wv,
        (c, k, k),
        (k - 1 - p, k - 1 - p),
        c,
        true,
    );

    let loss = |args: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        let y = mlx_rs::ops::conv2d(
            &args["x"],
            &wt,
            (1, 1),
            (p as i32, p as i32),
            None,
            c as i32,
        )?;
        Ok(vec![y.square()?.sum(None)?])
    };
    let mut bad = 0;
    let mut worst = 0f32;
    for run in 0..200 {
        // Dirty the free list with NaNs the size of every buffer this graph allocates.
        let junk: Vec<Array> = (0..8)
            .map(|i| {
                Array::from_slice(
                    &vec![f32::NAN; (n * h * w * c) << (i % 4)],
                    &[((n * h * w * c) << (i % 4)) as i32],
                )
            })
            .collect();
        mlx_rs::transforms::eval(junk.iter()).unwrap();
        drop(junk);

        let x = Array::from_slice(&xv, &[n as i32, h as i32, w as i32, c as i32]);
        let y = mlx_rs::ops::conv2d(&x, &wt, (1, 1), (p as i32, p as i32), None, c as i32).unwrap();
        let mut vg = keyed_value_and_grad(loss);
        let mut args = HashMap::new();
        args.insert(Rc::from("x"), x);
        let (_, g) = vg(args, 0).unwrap();
        let dx = &g["x"];
        mlx_rs::transforms::eval([&y, dx]).unwrap();
        let d = close(y.as_slice::<f32>(), &y_ref).max(close(dx.as_slice::<f32>(), &dx_ref));
        if d.is_nan() || d >= 1e-3 {
            bad += 1;
            worst = if d.is_nan() { f32::NAN } else { worst.max(d) };
            eprintln!("run {run}: max |Δ| {d:e}");
        }
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    assert_eq!(
        bad, 0,
        "{bad}/200 runs differ from the host reference (worst {worst:e})"
    );
}
