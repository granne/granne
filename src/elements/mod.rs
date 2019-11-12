use ordered_float::NotNan;

mod angular_vector;

pub use angular_vector::{AngularIntVector, AngularIntVectors, AngularVector, AngularVectors};

/// A trait for any type containing elements to be indexed using `GranneBuilder` and/or used for
/// searching with `Granne`.
pub trait ElementContainer {
    type Element;

    fn get(self: &Self, idx: usize) -> Self::Element;
    fn len(self: &Self) -> usize;
    fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32>;

    fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
        self.dist_to_element(i, &self.get(j))
    }

    fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
        let element = self.get(idx);
        others
            .iter()
            .map(|&j| self.dist_to_element(j, &element))
            .collect()
    }

    fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }
}

pub trait Dist<Other> {
    fn dist(self: &Self, other: &Other) -> NotNan<f32>;
}

impl<E: Dist<E> + Clone> ElementContainer for [E] {
    type Element = E;

    fn get(self: &Self, idx: usize) -> Self::Element {
        self[idx].clone()
    }

    fn len(self: &Self) -> usize {
        (*self).len()
    }

    fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
        self[idx].dist(element)
    }

    fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
        self[i].dist(&self[j])
    }

    fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
        others.iter().map(|&j| self[idx].dist(&self[j])).collect()
    }
}
