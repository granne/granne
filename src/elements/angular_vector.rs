use super::{Dist, ElementContainer, ExtendableElementContainer};
use crate::{io, slice_vector::FixedWidthSliceVector};

use ordered_float::NotNan;
use std::cmp;

use std::borrow::Cow;
use std::io::{Result, Write};
use std::iter::FromIterator;

#[derive(Clone)]
pub struct AngularVectorT<'a, T: Copy>(pub Cow<'a, [T]>);

impl<'a, T: Copy> AngularVectorT<'a, T> {
    pub fn len(self: &Self) -> usize {
        self.0.len()
    }

    pub fn into_owned(self: Self) -> AngularVectorT<'static, T> {
        AngularVectorT(self.0.into_owned().into())
    }

    pub fn to_vec(self: Self) -> Vec<T> {
        self.0.into_owned()
    }

    pub fn as_slice(self: &Self) -> &[T] {
        &self.0[..]
    }
}

pub type AngularVector<'a> = AngularVectorT<'a, f32>;
pub type AngularIntVector<'a> = AngularVectorT<'a, i8>;

impl From<Vec<f32>> for AngularVector<'static> {
    fn from(v: Vec<f32>) -> Self {
        // todo speedup
        let mut v = v;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm >= 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }

        Self(Cow::from(v))
    }
}

impl FromIterator<f32> for AngularVector<'static> {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let v: Vec<f32> = iter.into_iter().collect();
        Self::from(v)
    }
}

impl From<Vec<f32>> for AngularIntVector<'static> {
    fn from(v: Vec<f32>) -> Self {
        Self::quantize(&v)
    }
}

impl FromIterator<f32> for AngularIntVector<'static> {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let v: Vec<f32> = iter.into_iter().collect();
        Self::from(v)
    }
}

#[derive(Clone)]
/// A collection of `AngularVectorT`s
pub struct AngularVectorsT<'a, T: Copy>(FixedWidthSliceVector<'a, T>);

/// A collection of `AngularVector`s
pub type AngularVectors<'a> = AngularVectorsT<'a, f32>;

/// A collection of `AngularIntVector`s
pub type AngularIntVectors<'a> = AngularVectorsT<'a, i8>;

impl<'a, T: Copy> AngularVectorsT<'a, T> {
    /// Create a new collection vector. The dimension will be set once the first vector is pushed
    /// into the collection.
    pub fn new() -> Self {
        Self(FixedWidthSliceVector::new())
    }

    /// Load a collection of vectors with dimension `dim` from a `u8` buffer.
    /// `buffer` needs to contain ...
    pub fn load(buffer: &'a [u8], dim: usize) -> Self {
        Self(FixedWidthSliceVector::load(buffer, dim))
    }

    /// Create a collection of vectors with dimension `dim` from a `Vec<T>`.
    ///
    pub fn from_vec(vec: Vec<T>, dim: usize) -> Self {
        Self(FixedWidthSliceVector::with_data(vec, dim))
    }

    /// Borrows the data
    pub fn borrow(self: &'a Self) -> AngularVectorsT<'a, T> {
        Self(self.0.borrow())
    }

    /// Clones the underlying data if not already owned.
    pub fn into_owned(self: Self) -> AngularVectorsT<'static, T> {
        Self(self.0.into_owned())
    }

    pub fn extend(self: &mut Self, vec: AngularVectorsT<'_, T>) {
        self.0.extend_from_slice_vector(&vec.0)
    }

    /// Pushes `vec` onto the collection
    pub fn push(self: &mut Self, vec: &AngularVectorT<'_, T>) {
        self.0.push(&vec.0[..]);
    }

    /// Returns the number of vectors in this collection.
    pub fn len(self: &Self) -> usize {
        self.0.len()
    }

    /// Returns a reference to the vector at `index`.
    pub fn get_element(self: &'a Self, index: usize) -> AngularVectorT<'a, T> {
        AngularVectorT(Cow::Borrowed(self.0.get(index)))
    }
}

impl<'a, T: Copy> FromIterator<AngularVectorT<'a, T>> for AngularVectorsT<'static, T> {
    fn from_iter<I: IntoIterator<Item = AngularVectorT<'a, T>>>(iter: I) -> Self {
        let mut vecs = AngularVectorsT::new();
        for vec in iter {
            vecs.push(&vec);
        }

        vecs
    }
}

impl<'a, T: Copy> io::Writeable for AngularVectorsT<'a, T> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        self.0.write(buffer)
    }
}

macro_rules! element_container_impl {
    ($scalar_type:ty) => {
        impl<'a> ElementContainer for AngularVectorsT<'a, $scalar_type> {
            type Element = AngularVectorT<'static, $scalar_type>;

            fn get(self: &Self, idx: usize) -> Self::Element {
                self.get_element(idx).into_owned()
            }

            fn len(self: &Self) -> usize {
                self.len()
            }

            fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
                self.get_element(idx).dist(element)
            }

            fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
                self.get_element(i).dist(&self.get_element(j))
            }

            fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
                let element = self.get_element(idx);
                others
                    .iter()
                    .map(|&j| element.dist(&self.get_element(j)))
                    .collect()
            }
        }

        impl<'a> ExtendableElementContainer for AngularVectorsT<'a, $scalar_type> {
            type InternalElement = Self::Element;

            fn push(self: &mut Self, element: Self::InternalElement) {
                self.push(&element)
            }
        }
    };
}

element_container_impl!(f32);
element_container_impl!(i8);

impl<'a, 'b> Dist<AngularVector<'b>> for AngularVector<'a> {
    fn dist(self: &Self, other: &AngularVector<'b>) -> NotNan<f32> {
        // optimized code to compute r for systems supporting avx2
        // with fallback for other systems
        // see https://doc.rust-lang.org/std/arch/index.html#examples
        #[inline(always)]
        fn compute_r(x: &[f32], y: &[f32]) -> f32 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return unsafe { compute_r_avx2(x, y) };
                }
            }

            compute_r_fallback(x, y)
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn compute_r_avx2(x: &[f32], y: &[f32]) -> f32 {
            // this function will be inlined and take advantage of avx2 auto-vectorization
            compute_r_fallback(x, y)
        }

        #[inline(always)]
        fn compute_r_fallback(x: &[f32], y: &[f32]) -> f32 {
            const CHUNK_SIZE: usize = 8;
            let mut R = [0.0f32; CHUNK_SIZE];

            for (a, b) in x.chunks_exact(CHUNK_SIZE).zip(y.chunks_exact(CHUNK_SIZE)) {
                for i in 0..CHUNK_SIZE {
                    R[i] = a[i].mul_add(b[i], R[i]);
                }
            }

            let mut r = 0.0f32;
            for i in 0..CHUNK_SIZE {
                r += R[i];
            }

            for (ai, bi) in x
                .chunks_exact(CHUNK_SIZE)
                .remainder()
                .iter()
                .zip(y.chunks_exact(CHUNK_SIZE).remainder())
            {
                r = ai.mul_add(*bi, r);
            }

            r
        }

        let &AngularVectorT(ref x) = self;
        let &AngularVectorT(ref y) = other;

        // try with chunks_exact in order to force simd
        // https://rust.godbolt.org/z/G5A2u0
        // https://news.ycombinator.com/item?id=21342501
        /*        let mut r = 0.0f32;
        for (x, y) in x.iter().zip(y.iter()) {
            r = x.mul_add(*y, r);
        }
        */

        let r = compute_r(x, y);

        let d = NotNan::new(1.0f32 - r).unwrap();

        cmp::max(0.0f32.into(), d)
    }
}

const MAX_QVALUE: f32 = 127.0;

impl AngularIntVector<'static> {
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

impl<'a, 'b> Dist<AngularIntVector<'b>> for AngularIntVector<'a> {
    fn dist(self: &Self, other: &AngularIntVector<'b>) -> NotNan<f32> {
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

        let &AngularVectorT(ref x) = self;
        let &AngularVectorT(ref y) = other;

        let (r, dx, dy) = compute_r_dx_dy(x, y);

        let r =
            NotNan::new(r / (dx.sqrt() * dy.sqrt())).unwrap_or_else(|_| NotNan::new(0.0).unwrap());
        let d = NotNan::new(1.0f32).unwrap() - r;

        cmp::max(NotNan::new(0.0f32).unwrap(), d)
    }
}

#[allow(unused)]
pub fn angular_reference_dist(first: &AngularVector, second: &AngularVector) -> NotNan<f32> {
    let &AngularVectorT(ref x) = first;
    let &AngularVectorT(ref y) = second;

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
            let x: AngularVector = test_helper::random_floats().take(100).collect();
            let y: AngularVector = test_helper::random_floats().take(100).collect();

            assert!((x.dist(&y) - angular_reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x: AngularVector = test_helper::random_floats().take(100).collect();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_opposite_vector() {
        for _ in 0..100 {
            let x: AngularVector = test_helper::random_floats().take(100).collect();
            let y: AngularVector = x.0.clone().into_iter().map(|x| -x).collect();

            assert!(x.dist(&y).into_inner() > 2.0f32 - DIST_EPSILON);
        }
    }

    #[test]
    fn test_array() {
        let a: AngularVector = vec![0f32, 1f32, 2f32].into_iter().collect();

        a.dist(&a);
    }

    #[test]
    fn test_large_arrays() {
        let x = vec![1.0f32; 100];

        let a: AngularVector = x.into_iter().collect();

        a.dist(&a);
    }
}
