use std::cmp;
use ordered_float::NotNaN;
use rblas;

use super::{ComparableTo, DIM, NormalizedFloatElement};

#[repr(C)]
#[derive(Clone)]
pub struct Int8Element(pub [i8; DIM]);

const INT8_ELEMENT_NORM: i32 = 100;


impl ComparableTo<Int8Element> for Int8Element {
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
        let NormalizedFloatElement(ref element) = element;

        let mut array = [0i8; DIM];
        for i in 0..DIM {
            array[i] = (element[i] * INT8_ELEMENT_NORM as f32).round() as i8;
        }

        array.into()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::example;

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
