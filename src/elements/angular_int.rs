//! This module contains element types for quantized angular vectors using `i8` as scalars.
//!
//! An [`angular::Vector`](../angular/struct.Vector.html) is converted into an
//! [`angular_int::Vector`](struct.Vector.html) by mapping each dimension (originally stored as
//! `f32`) into the range [-127, 127], which is then stored as `i8`, saving 3 bytes per dimension.

use super::{Dist, ElementContainer, ExtendableElementContainer};
use crate::{io, math, slice_vector::FixedWidthSliceVector};

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
        let &Vector(ref x) = self;
        let &Vector(ref y) = other;

        let (r, dx, dy) = math::dot_product_and_squared_norms_i8(x, y);
        let (r, dx, dy) = (r as f32, dx as f32, dy as f32);

        let r = NotNan::new(r / (dx.sqrt() * dy.sqrt())).unwrap_or_else(|_| NotNan::new(0.0).unwrap());
        let d = NotNan::new(1.0f32).unwrap() - r;

        cmp::max(NotNan::new(0.0f32).unwrap(), d)
    }
}
