use std::cmp;
use ordered_float::NotNaN;
use rblas;

pub const DIM: usize = 100;
pub const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

pub trait HasDistance {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32>;
}

#[repr(C)]
#[derive(Clone)]
pub struct FloatElement([f32; DIM]);

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
    pub fn normalized(self: FloatElement) -> NormalizedFloatElement {

        let FloatElement(mut unnormed) = self;
        let norm: f32 = rblas::Nrm2::nrm2(&unnormed[..]);
        rblas::Scal::scal(&(1.0 / norm), &mut unnormed[..]);

        NormalizedFloatElement(unnormed)
    }
}


impl HasDistance for FloatElement {
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


#[repr(C)]
#[derive(Clone)]
pub struct NormalizedFloatElement([f32; DIM]);

impl From<[f32; DIM]> for NormalizedFloatElement {
    fn from(array: [f32; DIM]) -> NormalizedFloatElement {
        NormalizedFloatElement(array)
    }
}

impl HasDistance for NormalizedFloatElement {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &NormalizedFloatElement(ref x) = self;
        let &NormalizedFloatElement(ref y) = other;

        let r: f32 = rblas::Dot::dot(&x[..], &y[..]);

        let d = NotNaN::new(1.0f32 - r).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
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


#[repr(C)]
#[derive(Clone)]
pub struct Int8Element(pub [i8; DIM]);

const INT8_ELEMENT_NORM: i32 = 100;

impl From<[i8; DIM]> for Int8Element {
    fn from(array: [i8; DIM]) -> Int8Element {
        Int8Element(array)
    }
}

impl<'a> From<&'a [i8]> for Int8Element {
    fn from(slice: &'a [i8]) -> Int8Element {
        assert_eq!(DIM, slice.len());

        let mut array = [0i8; DIM];
        array.copy_from_slice(slice);

        array.into()
    }
}

impl From<Vec<i8>> for Int8Element {
    fn from(vec: Vec<i8>) -> Int8Element {
        vec.as_slice().into()
    }
}

impl PartialEq for Int8Element {
    fn eq(self: &Self, other: &Self) -> bool {
        let &Int8Element(ref x) = self;
        let &Int8Element(ref y) = other;

        x.iter().zip(y.iter()).all(|(x, y)| x == y)
    }
}


impl From<NormalizedFloatElement> for Int8Element {
    fn from(element: NormalizedFloatElement) -> Int8Element {
        let NormalizedFloatElement(element) = element;

        let mut array = [0i8; DIM];
        for i in 0..DIM {
            array[i] = (element[i] * INT8_ELEMENT_NORM as f32).round() as i8;
        }

        array.into()
    }
}


impl HasDistance for Int8Element {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32> {
        let &Int8Element(ref x) = self;
        let &Int8Element(ref y) = other;

        let r: i32 = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi as i32 * yi as i32)
            .sum();

        const INT8_ELEMENT_NORM_SQUARED: f32 = (INT8_ELEMENT_NORM * INT8_ELEMENT_NORM) as f32;

        let d = NotNaN::new(1.0f32 - (r as f32 / INT8_ELEMENT_NORM_SQUARED)).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


pub mod example {
    use super::*;
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

    pub fn random_int8_element() -> Int8Element {
        random_float_element().normalized().into()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
    fn int8_dist() {
        for _ in 0..100 {
            let x: NormalizedFloatElement = example::random_float_element().normalized();
            let y: NormalizedFloatElement = example::random_float_element().normalized();

            let xi8: Int8Element = x.clone().into();
            let yi8: Int8Element = y.clone().into();

            // looser condition since conversion into Int8Elements
            // causes quantization effects
            assert!((xi8.dist(&yi8) - x.dist(&y)).abs() < 0.02);
        }
    }

    #[test]
    fn dist_between_same_vector() {
        for _ in 0..100 {
            let x = example::random_float_element();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }

    #[test]
    fn into_int8_element() {
        let vec: Vec<i8> = (-50..50).collect();
        let mut array = [0i8; 100];
        array.copy_from_slice(vec.as_slice());

        let array_element: Int8Element = array.into();
        let slice_element: Int8Element = vec.as_slice().into();
        let vec_element: Int8Element = vec.into();

        assert!(vec_element == array_element);
        assert!(slice_element == array_element);
    }
}
