use std::cmp;
use ordered_float::NotNaN;
use rblas;
use std::ops::Deref;

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

        AngularVector(data)
    }
}


impl<D> ComparableTo<Self> for AngularVector<D> 
    where D: Array<f32>
{
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        compute_distance(self.0.as_slice(), other.0.as_slice())
    }
}


impl<D> ComparableTo<[f32]> for AngularVector<D>
    where D: Array<f32>
{
    fn dist(self: &Self, other: &[f32]) -> NotNaN<f32> {
        assert_eq!(self.0.as_slice().len(), other.len());
        let norm: f32 = rblas::Nrm2::nrm2(&other[..]);

        compute_distance(self.0.as_slice(), &other)
    }
}

impl<D> Dense for AngularVector<D> 
    where D: Array<f32> 
{
    fn dim(self: &Self) -> usize {
        self.0.as_slice().len()
    }
}


/*
impl<T, A: Unsize<[T]>> Array<T> for A {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}
*/

#[inline(always)]
fn compute_distance(x: &[f32], y: &[f32]) -> NotNaN<f32> {
    let r: f32 = rblas::Dot::dot(x, y);

    let d = NotNaN::new(1.0f32 - r).unwrap();

    cmp::max(0f32.into(), d)
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

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

    #[test]
    fn test_vector() {
        let x = [1.0f32; 3];
        let v = vec![1f32, 2f32, 3f32];

        let a: AngularVector<[f32; 3]> = x.into();

        a.dist(&v[..]);
    }
}
