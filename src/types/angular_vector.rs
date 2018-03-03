use std::cmp;
use ordered_float::NotNaN;
use rblas;

use std::iter::FromIterator;

use super::{ComparableTo, Dense};
use super::array::Array;

#[repr(C)]
#[derive(Clone)]
pub struct AngularVector<D>(pub D)
    where D: Array<f32>;


impl<D> From<D> for AngularVector<D>
    where D: Array<f32>
{
    fn from(data: D) -> Self {
        let mut data = data;
        let norm: f32 = rblas::Nrm2::nrm2(data.as_slice());
        rblas::Scal::scal(&(1.0 / norm), data.as_mut_slice());

        AngularVector::<D>(data)
    }
}


impl<D> FromIterator<f32> for AngularVector<D>
    where D: Array<f32>
{
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut data: D = unsafe { ::std::mem::uninitialized() };
        let mut iter = iter.into_iter();
        for x in data.as_mut_slice() {
            *x = iter.next().expect("Too few elements");
        }
        assert_eq!(0, iter.count(), "Too many elements");

        data.into()
    }
}


impl<D> ComparableTo<Self> for AngularVector<D>
    where D: Array<f32>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        compute_distance(self.0.as_slice(), other.0.as_slice())
    }
}


impl<D> Dense<f32> for AngularVector<D>
    where D: Array<f32>
{
    fn dim() -> usize {
        ::std::mem::size_of::<D>() / ::std::mem::size_of::<f32>()
    }

    fn as_slice(self: &Self) -> &[f32] {
        self.0.as_slice()
    }
}

impl<D> Dense<i8> for AngularIntVector<D>
    where D: Array<i8>
{
    fn dim() -> usize {
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


impl<D, T> From<AngularVector<D>> for AngularIntVector<T>
    where D: Array<f32>,
          T: Array<i8>
{
    fn from(element: AngularVector<D>) -> AngularIntVector<T> {
        let AngularVector(ref element) = element;
        let element = element.as_slice();
        let mut array: T = unsafe { ::std::mem::uninitialized() };
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
    let r: f32 = rblas::Dot::dot(x, y);

    let d = NotNaN::new(1.0f32 - r).unwrap();

    cmp::max(0f32.into(), d)
}


pub fn angular_reference_dist<D: Array<f32>>(first: &AngularVector<D>, second: &AngularVector<D>) -> NotNaN<f32> {
    let &AngularVector::<D>(ref x) = first;
    let &AngularVector::<D>(ref y) = second;

    let r: f32 = x.as_slice().iter()
        .zip(y.as_slice().iter())
        .map(|(&xi, &yi)| xi as f32 * yi as f32)
        .sum();

    let dx: f32 = x.as_slice().iter().map(|&xi| xi as f32 * xi as f32).sum();
    let dy: f32 = y.as_slice().iter().map(|&yi| yi as f32 * yi as f32).sum();

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
            let x: AngularVector<[f32; 100]> = example::random_dense_element();
            let y: AngularVector<[f32; 100]> = example::random_dense_element();

            assert!((x.dist(&y) - angular_reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x: AngularVector<[f32; 100]> = example::random_dense_element();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }

    #[test]
    fn dist_between_opposite_vector() {
        for _ in 0..100 {
            let x: AngularVector<[f32; 100]> = example::random_dense_element();
            let y: AngularVector<[f32; 100]> =
                x.0.clone().into_iter().map(|x| -x).collect();

            assert!(x.dist(&y).into_inner() > 2.0f32 - DIST_EPSILON);
        }
    }

    #[test]
    fn test_array() {
        let a: AngularVector<[f32; 3]> = [0f32, 1f32, 2f32].into();

        a.dist(&a);
    }

    #[test]
    fn test_large_arrays() {
        let x = [1.0f32; 100];

        let a: AngularVector<[f32; 100]> = x.into();

        a.dist(&a);
    }
}
