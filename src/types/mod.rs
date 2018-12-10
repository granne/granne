use ordered_float::NotNaN;
use std::iter::FromIterator;

mod angular_vector;
mod mmap;

pub use self::angular_vector::*;
pub use self::mmap::*;

#[cfg(test)]
pub const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

pub trait ComparableTo<B: ?Sized> {
    fn dist(self: &Self, other: &B) -> NotNaN<f32>;
    fn eps() -> NotNaN<f32> {
        NotNaN::new(0.0000001f32).unwrap()
    }
}

pub trait Dense<T> {
    fn dim(self: &Self) -> usize;

    fn as_slice(self: &Self) -> &[T];
}

pub mod example {
    use super::*;
    use rand;
    use rand::Rng;

    pub fn random_dense_element<T>(dim: usize) -> T
        where T: Dense<f32> + FromIterator<f32>
    {
        let mut rng = rand::thread_rng();

        let mut data: Vec<f32> = Vec::new();

        for _ in 0..dim {
            data.push(rng.gen::<f32>() - 0.5);
        }

        data.into_iter().collect()
    }
}
