//! This module contains element types for angular vectors using `f32` as scalars.

use super::{Dist, ElementContainer, ExtendableElementContainer};
use crate::{io, math, slice_vector::FixedWidthSliceVector};

use ordered_float::NotNan;
use std::cmp;

use std::borrow::Cow;
use std::io::{Result, Write};
use std::iter::FromIterator;

dense_vector!(f32);

impl From<Vec<f32>> for Vector<'static> {
    fn from(mut v: Vec<f32>) -> Self {
        /*        // todo speedup
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm >= 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }

        */
        math::normalize_f32(&mut v);

        Self(Cow::from(v))
    }
}

impl<'a, 'b> Dist<Vector<'b>> for Vector<'a> {
    fn dist(self: &Self, other: &Vector<'b>) -> NotNan<f32> {
        let &Vector(ref x) = self;
        let &Vector(ref y) = other;

        let r = math::dot_product_f32(x, y);

        let d = NotNan::new(1.0f32 - r).unwrap();

        cmp::max(0.0f32.into(), d)
    }
}

#[doc(hidden)]
#[allow(unused)]
pub fn angular_reference_dist(first: &Vector, second: &Vector) -> NotNan<f32> {
    let &Vector(ref x) = first;
    let &Vector(ref y) = second;

    let r: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi as f32 * yi as f32)
        .sum();

    let dx: f32 = x.iter().map(|&xi| xi as f32 * xi as f32).sum();
    let dy: f32 = y.iter().map(|&yi| yi as f32 * yi as f32).sum();

    let d = NotNan::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

    cmp::max(NotNan::new(0.0f32).unwrap(), d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

    #[test]
    fn reference_dist() {
        for _ in 0..100 {
            let x: Vector = test_helper::random_floats().take(100).collect();
            let y: Vector = test_helper::random_floats().take(100).collect();

            assert!((x.dist(&y) - angular_reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x: Vector = test_helper::random_floats().take(100).collect();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_opposite_vector() {
        for _ in 0..100 {
            let x: Vector = test_helper::random_floats().take(100).collect();
            let y: Vector = x.0.clone().into_iter().map(|x| -x).collect();

            assert!(x.dist(&y).into_inner() > 2.0f32 - DIST_EPSILON);
        }
    }

    #[test]
    fn test_array() {
        let a: Vector = vec![0f32, 1f32, 2f32].into_iter().collect();

        a.dist(&a);
    }

    #[test]
    fn test_large_arrays() {
        let x = vec![1.0f32; 100];

        let a: Vector = x.into_iter().collect();

        a.dist(&a);
    }
}
