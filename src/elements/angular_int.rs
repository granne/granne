use super::{Dist, ElementContainer, ExtendableElementContainer};
use crate::{io, slice_vector::FixedWidthSliceVector};

use super::dense_vector;

use ordered_float::NotNan;
use std::cmp;

use std::borrow::Cow;
use std::io::{Result, Write};
use std::iter::FromIterator;

dense_vector!(i8);

impl From<Vec<f32>> for Vector<'static> {
    fn from(v: Vec<f32>) -> Self {
        Self::quantize(&v)
    }
}

const MAX_QVALUE: f32 = 127.0;

impl Vector<'static> {
    fn quantize(s: &[f32]) -> Self {
        let max_value = s
            .iter()
            .map(|s| NotNan::new(s.abs()).unwrap())
            .max()
            .unwrap_or_else(|| NotNan::new(MAX_QVALUE).unwrap());

        let mut v = Vec::with_capacity(s.len());

        for x in s {
            let vi = x * MAX_QVALUE / *max_value;
            debug_assert!(-MAX_QVALUE - 0.0001 <= vi && vi <= MAX_QVALUE + 0.0001);
            v.push(vi as i8);
        }

        Self(Cow::from(v))
    }
}

impl<'a, 'b> Dist<Vector<'b>> for Vector<'a> {
    fn dist(self: &Self, other: &Vector<'b>) -> NotNan<f32> {
        // optimized code to compute r, dx, and dy for systems supporting avx2
        // with fallback for other systems
        // see https://doc.rust-lang.org/std/arch/index.html#examples
        fn compute_r_dx_dy(x: &[i8], y: &[i8]) -> (f32, f32, f32) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { compute_r_dx_dy_avx2(x, y) };
                }
            }

            compute_r_dx_dy_fallback(x, y)
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        unsafe fn compute_r_dx_dy_avx2(x: &[i8], y: &[i8]) -> (f32, f32, f32) {
            // this function will be inlined and take advantage of avx2 auto-vectorization
            compute_r_dx_dy_fallback(x, y)
        }

        #[inline(always)]
        fn compute_r_dx_dy_fallback(x: &[i8], y: &[i8]) -> (f32, f32, f32) {
            let mut r = 0i32;
            let mut dx = 0i32;
            let mut dy = 0i32;

            for (xi, yi) in x
                .iter()
                .map(|&xi| i32::from(xi))
                .zip(y.iter().map(|&yi| i32::from(yi)))
            {
                r += xi * yi;
                dx += xi * xi;
                dy += yi * yi;
            }

            (r as f32, dx as f32, dy as f32)
        }

        let &Vector(ref x) = self;
        let &Vector(ref y) = other;

        let (r, dx, dy) = compute_r_dx_dy(x, y);

        let r =
            NotNan::new(r / (dx.sqrt() * dy.sqrt())).unwrap_or_else(|_| NotNan::new(0.0).unwrap());
        let d = NotNan::new(1.0f32).unwrap() - r;

        cmp::max(NotNan::new(0.0f32).unwrap(), d)
    }
}
