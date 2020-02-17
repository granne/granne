use ordered_float::NotNan;

#[macro_use]
mod dense_vector;

pub mod angular;
pub mod angular_int;

pub mod embeddings;

/// A trait for any type containing elements to be indexed using `GranneBuilder` and/or used for
/// searching with `Granne`.
///
/// It should be noted that the provided default implementations of `dist` and `dists` are rarely
/// the most efficient ones and for improved performance it is recommended to provide specialized
/// implementations.
pub trait ElementContainer {
    /// The type of element, this `ElementContainer` contains.
    type Element;

    /// Returns the element with offset/id `idx`.
    fn get(self: &Self, idx: usize) -> Self::Element;

    /// Returns the number of elements.
    fn len(self: &Self) -> usize;

    /// Returns the distance between the element at `idx` and `element`.
    fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32>;

    /// Returns the distance between the elemests at `i` and `j`.
    fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
        self.dist_to_element(i, &self.get(j))
    }

    /// Does a batch computation of distances from `idx` to all elements in `others`.
    fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
        let element = self.get(idx);
        others.iter().map(|&j| self.dist_to_element(j, &element)).collect()
    }

    /// Returns `true` if the container contains no elements.
    fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }
}

/// A trait for `ElementContainer`s that can be extended with more elements
pub trait ExtendableElementContainer: ElementContainer {
    /// Internal representation of an element (can be the same as
    /// [`ElementContainer::Element`](trait.ElementContainer.html#associatedtype.Element))
    type InternalElement;

    /// Moves an element into the container
    fn push(self: &mut Self, element: Self::InternalElement);
}

/// A trait for `ElementContainer`s that can be permuted/reordered
pub trait Permutable {
    /// Reorder the elements such that the element at position `idx` is moved to `permutation[idx]`.
    fn permute(self: &mut Self, permutation: &[usize]);
}

/// `Dist<Other>` - A trait for types `E` and `Other` between which a distance can be computed.
///
/// By implementing `Dist<E>` for a type `E` one gets the `ElementContainer` trait implemented for
/// slices and `Vec`s of `E`, i.e., `[E]: ElementContainer` and `Vec<E>: ElementContainer`
pub trait Dist<Other> {
    /// Returns the distance between `self` and `other`
    fn dist(self: &Self, other: &Other) -> NotNan<f32>;
}

impl<E: Dist<E> + Clone> ElementContainer for &[E] {
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

impl<E: Dist<E> + Clone> ElementContainer for Vec<E> {
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

impl<Elements: ElementContainer> ElementContainer for &Elements {
    type Element = Elements::Element;

    fn get(self: &Self, idx: usize) -> Self::Element {
        Elements::get(self, idx)
    }

    fn len(self: &Self) -> usize {
        Elements::len(self)
    }

    fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
        Elements::dist_to_element(self, idx, element)
    }

    fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
        Elements::dist(self, i, j)
    }

    fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
        Elements::dists(self, idx, others)
    }
}
