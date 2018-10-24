use std::cmp;
use hnsw::{At,Writeable};
use ordered_float::NotNaN;
use blas;

use std::borrow::Cow;
use std::iter::FromIterator;
use std::io::{Write, Result};

use super::{ComparableTo, Dense};
use file_io;

#[derive(Clone)]
pub struct AngularVectorT<'a, T: Copy + 'static>(Cow<'a, [T]>);

impl<'a, T: Copy> AngularVectorT<'a, T> {
    fn len(self: &Self) -> usize {
        self.0.len()
    }
}

impl<T> FromIterator<f32> for AngularVectorT<'static, T>
    where T: Copy,
          AngularVectorT<'static, T>: From<Vec<f32>>
{
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let vec: Vec<f32> = iter.into_iter().collect();
        vec.into()
    }
}

pub type AngularVector<'a> = AngularVectorT<'a, f32>;
pub type AngularIntVector<'a> = AngularVectorT<'a, i8>;


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

impl<'a> Into<Vec<f32>> for AngularVector<'a> {
    fn into(self: Self) -> Vec<f32> {
        self.0.into_owned()
    }
}

impl<'a> ComparableTo<Self> for AngularVector<'a>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        assert_eq!(self.len(), other.len());

        let &AngularVectorT(ref x) = self;
        let &AngularVectorT(ref y) = other;

        let r: f32 = unsafe { blas::sdot(x.len() as i32, x, 1, y, 1) };

        let d = NotNaN::new(1.0f32 - r).unwrap();

        cmp::max(0f32.into(), d)
    }
}


impl<'a> Dense<f32> for AngularVector<'a>
{
    fn dim(self: &Self) -> usize {
        self.len()
    }

    fn as_slice(self: &Self) -> &[f32] {
        &self.0[..]
    }
}


#[derive(Clone)]
pub struct AngularVectorsT<'a, T: Copy + 'static> {
    data: Cow<'a, [T]>,
    pub dim: usize
}

pub type AngularVectors<'a> = AngularVectorsT<'a, f32>;
pub type AngularIntVectors<'a> = AngularVectorsT<'a, i8>;

impl<'a, T: Copy> AngularVectorsT<'a, T> {
    pub fn new(dimension: usize) -> Self {
        Self {
            data: Vec::new().into(),
            dim: dimension
        }
    }

    pub fn load(dim: usize, buffer: &'a [u8]) -> Self {
        let data: &[T] = file_io::load(buffer);

        assert_eq!(0, data.len() % dim);

        Self {
            data: data.into(),
            dim: dim
        }
    }

    pub fn from_vec(dim: usize, vec: Vec<T>) -> Self {
        assert_eq!(0, vec.len() % dim);

        Self {
            data: vec.into(),
            dim: dim
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
        AngularVectorT(Cow::Borrowed(&self.data[index*self.dim..(index+1)*self.dim]))
    }

    pub fn data(self: &'a Self) -> &'a [T] {
        &self.data[..]
    }
}


impl<'a, T: Copy> FromIterator<AngularVectorT<'a, T>> for AngularVectorsT<'static, T>
{
    fn from_iter<I: IntoIterator<Item = AngularVectorT<'a, T>>>(iter: I) -> Self {
        let mut vecs = AngularVectorsT::new(0);
        for vec in iter {
            vecs.push(&vec);
        }

        vecs
    }
}


// TODO: fix whenever HKT arrive
// See https://github.com/rust-lang/rfcs/blob/master/text/0235-collections-conventions.md#lack-of-iterator-methods
impl<'a, T: Copy + 'static> At for AngularVectorsT<'a, T>
{
    type Output=AngularVectorT<'static, T>;

    fn at(self: &Self, index: usize) -> Self::Output {
        AngularVectorT(self.data[index*self.dim..(index+1)*self.dim].to_vec().into())
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a> Writeable for AngularVectors<'a> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        file_io::write(&self.data[..], buffer)
    }
}


const MAX_QVALUE: usize = 127;

impl From<Vec<f32>> for AngularIntVector<'static> {
    fn from(vec: Vec<f32>) -> Self {
        let n = vec.len() as i32;
        let max_ind = unsafe { blas::isamax(n as i32, vec.as_slice(), 1)-1 };
        let max_value: f32 = vec[max_ind].abs();

        let mut vec = vec;
        if max_value > 0.0 {
            unsafe { blas::sscal(n, MAX_QVALUE as f32 / max_value, vec.as_mut_slice(), 1) };
        }

        AngularVectorT(vec.into_iter().map(|x| x as i8).collect::<Vec<i8>>().into())
    }
}

impl<'a> From<AngularVector<'a>> for AngularIntVector<'static> {
    fn from(vec: AngularVector<'a>) -> Self {
        vec.0.into_owned().into()
    }
}

impl<'a> ComparableTo<Self> for AngularIntVector<'a>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &AngularVectorT(ref x) = self;
        let &AngularVectorT(ref y) = other;

        let r = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi as i32 * yi as i32)
            .sum::<i32>() as f32;

        let dx = x.iter().map(|&xi| xi as i32 * xi as i32).sum::<i32>() as f32;
        let dy = y.iter().map(|&yi| yi as i32 * yi as i32).sum::<i32>() as f32;

        let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


pub fn angular_reference_dist(first: &AngularVector, second: &AngularVector) -> NotNaN<f32> {
    let &AngularVectorT(ref x) = first;
    let &AngularVectorT(ref y) = second;

    let r: f32 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi as f32 * yi as f32)
        .sum();

    let dx: f32 = x.iter().map(|&xi| xi as f32 * xi as f32).sum();
    let dy: f32 = y.iter().map(|&yi| yi as f32 * yi as f32).sum();

    let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

    cmp::max(NotNaN::new(0.0f32).unwrap(), d)
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

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
            let y: AngularVector =
                x.0.clone().into_iter().map(|x| -x).collect();

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
