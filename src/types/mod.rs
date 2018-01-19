use std::cmp;
use ordered_float::NotNaN;
use rblas;

mod float_element;
mod int8_element;

pub use self::float_element::*;
pub use self::int8_element::*;

pub const DIM: usize = 100;
pub const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

pub trait ComparableTo<B> {
    fn dist(self: &Self, other: &B) -> NotNaN<f32>;
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
