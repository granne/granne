use std::cmp;
use ordered_float::NotNaN;
use rblas;

use super::ComparableTo;

const DIM: usize = 100;

#[repr(C)]
#[derive(Clone)]
pub struct FloatElement(pub [f32; DIM]);


impl ComparableTo<FloatElement> for FloatElement {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &FloatElement(ref x) = self;
        let &FloatElement(ref y) = other;

        let r: f32 = rblas::Dot::dot(&x[..], &y[..]);
        let dx: f32 = rblas::Nrm2::nrm2(&x[..]);
        let dy: f32 = rblas::Nrm2::nrm2(&y[..]);

        let d = NotNaN::new(1.0f32 - (r / (dx * dy))).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


impl From<[f32; DIM]> for FloatElement {
    fn from(array: [f32; DIM]) -> FloatElement {
        FloatElement(array)
    }
}


impl<'a> From<&'a [f32]> for FloatElement {
    fn from(slice: &'a [f32]) -> FloatElement {
        assert_eq!(DIM, slice.len());

        let mut array = [0f32; DIM];
        array.copy_from_slice(slice);

        array.into()
    }
}


impl From<Vec<f32>> for FloatElement {
    fn from(vec: Vec<f32>) -> FloatElement {
        vec.as_slice().into()
    }
}


impl FloatElement {
    pub fn normalized(mut self: FloatElement) -> NormalizedFloatElement {

        let FloatElement(ref mut unnormed) = self;
        let norm: f32 = rblas::Nrm2::nrm2(&unnormed[..]);
        rblas::Scal::scal(&(1.0 / norm), &mut unnormed[..]);

        NormalizedFloatElement(*unnormed)
    }
}


#[repr(C)]
#[derive(Clone)]
pub struct NormalizedFloatElement(pub [f32; DIM]);


impl ComparableTo<NormalizedFloatElement> for NormalizedFloatElement {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &NormalizedFloatElement(ref x) = self;
        let &NormalizedFloatElement(ref y) = other;

        let r: f32 = rblas::Dot::dot(&x[..], &y[..]);

        let d = NotNaN::new(1.0f32 - r).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


impl From<[f32; DIM]> for NormalizedFloatElement {
    fn from(array: [f32; DIM]) -> NormalizedFloatElement {
        NormalizedFloatElement(array)
    }
}


pub fn reference_dist(first: &FloatElement, second: &FloatElement) -> NotNaN<f32> {
    let &FloatElement(x) = first;
    let &FloatElement(y) = second;

    let r: f32 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi as f32 * yi as f32)
        .sum();
    let dx: f32 = x.iter().map(|&xi| xi as f32 * xi as f32).sum();
    let dy: f32 = y.iter().map(|&yi| yi as f32 * yi as f32).sum();

    let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

    cmp::max(NotNaN::new(0.0f32).unwrap(), d)
}

use rand;
use rand::Rng;

pub fn random_float_element() -> FloatElement {
    let mut rng = rand::thread_rng();
    let mut data = [0.0f32; DIM];

    for f in &mut data[..] {
        *f = rng.gen();
    }
    data.into()
}




#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

    #[test]
    fn rblas_dist() {
        for _ in 0..100 {
            let x = example::random_float_element();
            let y = example::random_float_element();

            assert!((x.dist(&y) - reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn normed_dist() {
        for _ in 0..100 {
            let x: FloatElement = example::random_float_element();
            let y: FloatElement = example::random_float_element();

            let x_normed: NormalizedFloatElement = x.clone().normalized();
            let y_normed: NormalizedFloatElement = y.clone().normalized();

            assert!((x_normed.dist(&y_normed) - reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }


    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x = example::random_float_element();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }
}
