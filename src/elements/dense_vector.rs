use super::{Dist, ElementContainer, ExtendableElementContainer};
use crate::{io, slice_vector::FixedWidthSliceVector};

use ordered_float::NotNan;
use std::cmp;

use std::borrow::Cow;
use std::io::{Result, Write};
use std::iter::FromIterator;

macro_rules! dense_vector {
    ($vector_name:ident, $vectors_name:ident, $scalar_type:ty) => {
        #[derive(Clone)]
        pub struct $vector_name<'a>(pub Cow<'a, [$scalar_type]>);

        impl<'a> $vector_name<'a> {
            pub fn len(self: &Self) -> usize {
                self.0.len()
            }

            pub fn into_owned(self: Self) -> $vector_name<'static> {
                $vector_name(self.0.into_owned().into())
            }

            pub fn to_vec(self: Self) -> Vec<$scalar_type> {
                self.0.into_owned()
            }

            pub fn as_slice(self: &Self) -> &[$scalar_type] {
                &self.0[..]
            }
        }

        impl FromIterator<f32> for $vector_name<'static> {
            fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
                let v: Vec<f32> = iter.into_iter().collect();
                Self::from(v)
            }
        }

        #[derive(Clone)]
        /// A collection of `$vector_name`s
        pub struct $vectors_name<'a>(FixedWidthSliceVector<'a, $scalar_type>);

        impl<'a> $vectors_name<'a> {
            /// Create a new collection vector. The dimension will be set once the first vector is pushed
            /// into the collection.
            pub fn new() -> Self {
                Self(FixedWidthSliceVector::new())
            }

            /// Load a collection of vectors with dimension `dim` from a `u8` buffer.
            /// `buffer` needs to contain ...
            pub fn load(buffer: &'a [u8], dim: usize) -> Self {
                Self(FixedWidthSliceVector::load(buffer, dim))
            }

            /// Create a collection of vectors with dimension `dim` from a `Vec<$scalar_type>`.
            ///
            pub fn from_vec(vec: Vec<$scalar_type>, dim: usize) -> Self {
                Self(FixedWidthSliceVector::with_data(vec, dim))
            }

            /// Borrows the data
            pub fn borrow(self: &'a Self) -> $vectors_name<'a> {
                Self(self.0.borrow())
            }

            /// Clones the underlying data if not already owned.
            pub fn into_owned(self: Self) -> $vectors_name<'static> {
                Self(self.0.into_owned())
            }

            pub fn extend(self: &mut Self, vec: $vectors_name<'_>) {
                self.0.extend_from_slice_vector(&vec.0)
            }

            /// Pushes `vec` onto the collection
            pub fn push(self: &mut Self, vec: &$vector_name<'_>) {
                self.0.push(&vec.0[..]);
            }

            /// Returns the number of vectors in this collection.
            pub fn len(self: &Self) -> usize {
                self.0.len()
            }

            /// Returns a reference to the vector at `index`.
            pub fn get_element(self: &'a Self, index: usize) -> $vector_name<'a> {
                $vector_name(Cow::Borrowed(self.0.get(index)))
            }
        }

        impl<'a> FromIterator<$vector_name<'a>> for $vectors_name<'static> {
            fn from_iter<I: IntoIterator<Item = $vector_name<'a>>>(iter: I) -> Self {
                let mut vecs = $vectors_name::new();
                for vec in iter {
                    vecs.push(&vec);
                }

                vecs
            }
        }

        impl<'a> io::Writeable for $vectors_name<'a> {
            fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
                self.0.write(buffer)
            }
        }

        impl<'a> ElementContainer for $vectors_name<'a> {
            type Element = $vector_name<'static>;

            fn get(self: &Self, idx: usize) -> Self::Element {
                self.get_element(idx).into_owned()
            }

            fn len(self: &Self) -> usize {
                self.len()
            }

            fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
                self.get_element(idx).dist(element)
            }

            fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
                self.get_element(i).dist(&self.get_element(j))
            }

            fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
                let element = self.get_element(idx);
                others
                    .iter()
                    .map(|&j| element.dist(&self.get_element(j)))
                    .collect()
            }
        }

        impl<'a> ExtendableElementContainer for $vectors_name<'a> {
            type InternalElement = Self::Element;

            fn push(self: &mut Self, element: Self::InternalElement) {
                self.push(&element)
            }
        }
    };
}
