use super::{Dist, ElementContainer};
use crate::io;

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
}
/*
impl<T> FromIterator<f32> for AngularVectorT<'static, T>
where
    T: Copy,
    AngularVectorT<'static, T>: From<Vec<f32>>,
{
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let vec: Vec<f32> = iter.into_iter().collect();
        vec.into()
    }
}
*/
pub type AngularVector<'a> = AngularVectorT<'a, f32>;
pub type AngularIntVector<'a> = AngularVectorT<'a, i8>;
/*
impl From<Vec<f32>> for AngularVector<'static> {
    fn from(vec: Vec<f32>) -> Self {
        let mut vec = vec;
        let n = vec.len() as i32;
        let norm = unsafe { blas::snrm2(n, vec.as_slice(), 1) };
        if norm > 0.0 {
            unsafe { blas::sscal(n, 1.0 / norm, vec.as_mut_slice(), 1) };
        }

        AngularVectorT(vec.into())
    }
}
 */
/*
impl<'a> Into<Vec<f32>> for AngularVector<'a> {
    fn into(self: Self) -> Vec<f32> {
        self.0.into_owned()
    }
}
*/
/*
impl<'a> Into<Vec<f32>> for AngularIntVector<'a> {
    fn into(self: Self) -> Vec<f32> {
        let float_vector: AngularVector = self.into();
        float_vector.into()
    }
}
*/
/*
impl<'a> ComparableTo<Self> for AngularVector<'a> {
    fn dist(self: &Self, other: &Self) -> NotNan<f32> {
        assert_eq!(self.len(), other.len());

        let &AngularVectorT(ref x) = self;
        let &AngularVectorT(ref y) = other;

        let r: f32 = unsafe { blas::sdot(x.len() as i32, x, 1, y, 1) };

        let d = NotNan::new(1.0f32 - r).unwrap();

        cmp::max(0f32.into(), d)
    }
}
 */

impl<'a, 'b> Dist<AngularVector<'b>> for AngularVector<'a> {
    fn dist(self: &Self, other: &AngularVector<'b>) -> NotNan<f32> {
        angular_reference_dist(self, other)
    }
}

#[derive(Clone, Default)]
pub struct AngularVectorsT<'a, T: Copy> {
    data: Cow<'a, [T]>,
    dim: usize,
}

pub type AngularVectors<'a> = AngularVectorsT<'a, f32>;
pub type AngularIntVectors<'a> = AngularVectorsT<'a, i8>;

impl<'a, T: Copy> AngularVectorsT<'a, T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new().into(),
            dim: 0,
        }
    }

    pub fn load(dim: usize, buffer: &'a [u8]) -> Self {
        let data: &[T] = unsafe { crate::io::load_bytes_as(buffer) };

        assert_eq!(0, data.len() % dim);

        Self {
            data: data.into(),
            dim,
        }
    }

    pub fn from_vec(dim: usize, vec: Vec<T>) -> Self {
        assert_eq!(0, vec.len() % dim);

        Self {
            data: vec.into(),
            dim,
        }
    }

    pub fn into_owned(self: Self) -> AngularVectorsT<'static, T> {
        AngularVectorsT {
            data: self.data.into_owned().into(),
            dim: self.dim,
        }
    }

    pub fn extend(self: &mut Self, vec: AngularVectorsT<T>) {
        assert_eq!(self.dim, vec.dim);

        self.data.to_mut().extend_from_slice(&vec.data[..]);
    }

    pub fn push(self: &mut Self, vec: &AngularVectorT<T>) {
        if self.dim == 0 {
            self.dim = vec.len();
        }

        assert_eq!(self.dim, vec.len());

        self.data.to_mut().extend_from_slice(&vec.0[..]);
    }

    pub fn len(self: &Self) -> usize {
        if self.dim > 0 {
            self.data.len() / self.dim
        } else {
            0
        }
    }

    pub fn get_element(self: &'a Self, index: usize) -> AngularVectorT<'a, T> {
        AngularVectorT(Cow::Borrowed(
            &self.data[index * self.dim..(index + 1) * self.dim],
        ))
    }

    pub fn data(self: &'a Self) -> &'a [T] {
        &self.data[..]
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

impl<'a> ElementContainer for AngularVectors<'a> {
    type Element = AngularVector<'static>;

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

impl<'a> ElementContainer for AngularIntVectors<'a> {
    type Element = AngularIntVector<'static>;

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
/*
impl<'a, T: Copy> Writeable for AngularVectorsT<'a, T> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        buffer.write_all(crate::io::load_as_bytes(self.data.as_slice()))

    }
}

impl<'a, T: Copy> Appendable for AngularVectorsT<'a, T> {
    type Element = AngularVectorT<'static, T>;

    fn new() -> Self {
        Self::new(0)
    }

    fn append(self: &mut Self, element: AngularVectorT<T>) {
        self.push(&element);
    }
}
*/
const MAX_QVALUE: usize = 127;
/*
impl From<Vec<f32>> for AngularIntVector<'static> {
    fn from(vec: Vec<f32>) -> Self {
        let n = vec.len() as i32;
        let max_ind = unsafe { blas::isamax(n as i32, vec.as_slice(), 1) - 1 };
        let max_value: f32 = vec[max_ind].abs();

        let mut vec = vec;
        if max_value > 0.0 {
            unsafe { blas::sscal(n, MAX_QVALUE as f32 / max_value, vec.as_mut_slice(), 1) };
        }

        AngularVectorT(vec.into_iter().map(|x| x as i8).collect::<Vec<i8>>().into())
    }
}
*/
/*
impl<'a> From<AngularVector<'a>> for AngularIntVector<'static> {
    fn from(vec: AngularVector<'a>) -> Self {
        vec.0.into_owned().into()
    }
}
*/
/*
impl<'a> From<AngularIntVector<'a>> for AngularVector<'static> {
    fn from(vec: AngularIntVector<'a>) -> Self {
        vec.0.iter().map(|&x| f32::from(x)).collect::<Vec<f32>>().into()
    }
}
*/
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
    use super::super::*;
    use super::*;

    #[test]
    fn rblas_dist() {
        for _ in 0..100 {
            let x: AngularVector = example::random_dense_element(100);
            let y: AngularVector = example::random_dense_element(100);

            assert!((x.dist(&y) - angular_reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x: AngularVector = example::random_dense_element(100);

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_opposite_vector() {
        for _ in 0..100 {
            let x: AngularVector = example::random_dense_element(100);
            let y: AngularVector = x.0.clone().into_iter().map(|x| -x).collect();

            assert!(x.dist(&y).into_inner() > 2.0f32 - DIST_EPSILON);
        }
    }

    #[test]
    fn test_array() {
        let a: AngularVector = vec![0f32, 1f32, 2f32].into();

        a.dist(&a);
    }

    #[test]
    fn test_large_arrays() {
        let x = vec![1.0f32; 100];

        let a: AngularVector = x.into();

        a.dist(&a);
    }
}
