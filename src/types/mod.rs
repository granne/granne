use std::cmp;
use ordered_float::NotNaN;
use std::iter::FromIterator;

mod array;
mod angular_vector;

pub use self::array::Array;

pub use self::angular_vector::*;

pub const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

pub trait ComparableTo<B: ?Sized> {
    fn dist(self: &Self, other: &B) -> NotNaN<f32>;
}

pub trait Dense<T> {
    fn dim() -> usize;

    fn as_slice(self: &Self) -> &[T];
}

pub mod example {
    use super::*;
    use rand;
    use rand::Rng;

    pub fn random_dense_element<T, K>() -> T
        where T: Dense<K> + FromIterator<f32>
    {
        let mut rng = rand::thread_rng();

        let mut data: Vec<f32> = Vec::new();

        for _ in 0..T::dim() {
            data.push(rng.gen());
        }

        data.into_iter().collect()
    }
}
