use std::cmp;
use hnsw::{At,Writeable};
use ordered_float::NotNaN;
use blas;


use std::borrow::{ToOwned, Cow};
use std::iter::FromIterator;
use std::io::{BufWriter, Read, Write, Result};

use super::{ComparableTo, Dense};
use super::array::Array;
use file_io;

#[derive(Clone)]
pub struct AngularVector<'a>(Cow<'a, [f32]>);

#[derive(Clone)]
pub struct AngularVectors<'a> {
    data: Cow<'a, [f32]>,
    dim: usize
}

impl<'a> AngularVectors<'a> {
    pub fn new() -> Self {
        Self {
            data: Vec::new().into(),
            dim: 0
        }
    }

    pub fn load(dim: usize, buffer: &'a [u8]) -> Self {
        let data: &[f32] = file_io::load(buffer);

        assert_eq!(0, data.len() % dim);

        Self {
            data: data.into(),
            dim: dim
        }
    }

    pub fn from_vec(dim: usize, vec: Vec<f32>) -> Self {
        assert_eq!(0, vec.len() % dim);

        Self {
            data: vec.into(),
            dim: dim
        }
    }
    
    pub fn push(self: &mut Self, vec: &[f32]) {
        if self.dim == 0 {
            self.dim = vec.len();
        }

        assert_eq!(self.dim, vec.len());

        self.data.to_mut().extend_from_slice(vec);
    }

    pub fn len(self: &Self) -> usize {
        if self.dim > 0 {
            self.data.len() / self.dim
        } else {
            0
        }
    }
}


impl<'a> FromIterator<AngularVector<'a>> for AngularVectors<'static>
{
    fn from_iter<T: IntoIterator<Item = AngularVector<'a>>>(iter: T) -> Self {
        let mut vecs = AngularVectors::new();
        for vec in iter {
            vecs.push(&vec.0[..]);
        }
        
        vecs
    }
}


// TODO: fix
impl<'a> At for AngularVectors<'a> {
    type Output=AngularVector<'static>;

    fn at(self: &Self, index: usize) -> Self::Output {
        self.data[index*self.dim..(index+1)*self.dim].to_vec().into()
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

impl<'a> AngularVector<'a> {
    fn len(self: &Self) -> usize {
        self.0.len()
    }
    
}

impl From<Vec<f32>> for AngularVector<'static> {
    fn from(vec: Vec<f32>) -> Self {
        let mut vec = vec;
        let n = vec.len() as i32;
        let norm = unsafe { blas::snrm2(n, vec.as_slice(), 1) };
        if norm > 0.0 {
            unsafe { blas::sscal(n, 1.0 / norm, vec.as_mut_slice(), 1) };
        }

        AngularVector(vec.into())
    }
}

impl FromIterator<f32> for AngularVector<'static>
{
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let vec: Vec<f32> = iter.into_iter().collect();
        vec.into()
    }
}


impl<'a> ComparableTo<Self> for AngularVector<'a>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        compute_distance(&self.0, &other.0)
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

impl<D> Dense<i8> for AngularIntVector<D>
    where D: Array<i8>
{
    fn dim(self: &Self) -> usize {
        ::std::mem::size_of::<D>() / ::std::mem::size_of::<i8>()
    }

    fn as_slice(self: &Self) -> &[i8] {
        self.0.as_slice()
    }
}


#[repr(C)]
#[derive(Clone)]
pub struct AngularIntVector<D>(pub D)
    where D: Array<i8>;

const INT8_ELEMENT_NORM: i32 = 100;

impl<D> From<D> for AngularIntVector<D>
    where D: Array<i8>
{
    fn from(data: D) -> Self {
        AngularIntVector::<D>(data)
    }
}


impl<D> FromIterator<i32> for AngularIntVector<D>
    where D: Array<i8>
{
    fn from_iter<T: IntoIterator<Item = i32>>(iter: T) -> Self {
        let mut data: D = unsafe { ::std::mem::uninitialized() };
        let mut iter = iter.into_iter();
        for x in data.as_mut_slice() {
            *x = iter.next().expect("Too few elements") as i8;
        }
        assert_eq!(0, iter.count(), "Too many elements");

        data.into()
    }
}


impl<D> From<AngularVector<'static>> for AngularIntVector<D>
    where D: Array<i8>
{
    fn from(element: AngularVector<'static>) -> AngularIntVector<D> {
        let AngularVector(ref element) = element;
        let element = &element[..];
        let mut array: D = unsafe { ::std::mem::uninitialized() };
        {
            let array = array.as_mut_slice();

            assert_eq!(array.len(), element.len());

            for i in 0..array.len() {
                array[i] = (element[i] * INT8_ELEMENT_NORM as f32).round() as i8;
            }
        }

        array.into()
    }
}


impl<D> ComparableTo<Self> for AngularIntVector<D>
    where D: Array<i8>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &AngularIntVector(ref x) = self;
        let &AngularIntVector(ref y) = other;

        let r: i32 = x.as_slice().iter()
            .zip(y.as_slice().iter())
            .map(|(&xi, &yi)| xi as i32 * yi as i32)
            .sum();

        const INT8_ELEMENT_NORM_SQUARED: f32 = (INT8_ELEMENT_NORM * INT8_ELEMENT_NORM) as f32;

        let d = NotNaN::new(1.0f32 - (r as f32 / INT8_ELEMENT_NORM_SQUARED)).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


#[inline(always)]
fn compute_distance(x: &[f32], y: &[f32]) -> NotNaN<f32> {
    let r: f32 = unsafe { blas::sdot(x.len() as i32, x, 1, y, 1) };

    let d = NotNaN::new(1.0f32 - r).unwrap();

    cmp::max(0f32.into(), d)
}


pub fn angular_reference_dist(first: &AngularVector, second: &AngularVector) -> NotNaN<f32> {
    let &AngularVector(ref x) = first;
    let &AngularVector(ref y) = second;

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
