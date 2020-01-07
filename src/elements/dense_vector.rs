macro_rules! dense_vector {
    ($scalar_type:ty) => {
        #[derive(Clone)]
        pub struct Vector<'a>(pub Cow<'a, [$scalar_type]>);

        impl<'a> Vector<'a> {
            pub fn len(self: &Self) -> usize {
                self.0.len()
            }

            pub fn into_owned(self: Self) -> Vector<'static> {
                Vector(self.0.into_owned().into())
            }

            pub fn to_vec(self: Self) -> Vec<$scalar_type> {
                self.0.into_owned()
            }

            pub fn as_slice(self: &Self) -> &[$scalar_type] {
                &self.0[..]
            }
        }

        impl FromIterator<f32> for Vector<'static> {
            fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
                let v: Vec<f32> = iter.into_iter().collect();
                Self::from(v)
            }
        }

        #[derive(Clone)]
        /// A collection of `Vector`s
        pub struct Vectors<'a>(FixedWidthSliceVector<'a, $scalar_type>);

        impl<'a> Vectors<'a> {
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

            /// Create a collection of vectors with dimension `dim` from a `Vec`.
            ///
            pub fn from_vec(vec: Vec<$scalar_type>, dim: usize) -> Self {
                Self(FixedWidthSliceVector::with_data(vec, dim))
            }

            /// Borrows the data
            pub fn borrow(self: &'a Self) -> Vectors<'a> {
                Self(self.0.borrow())
            }

            /// Clones the underlying data if not already owned.
            pub fn into_owned(self: Self) -> Vectors<'static> {
                Self(self.0.into_owned())
            }

            pub fn extend(self: &mut Self, vec: Vectors<'_>) {
                self.0.extend_from_slice_vector(&vec.0)
            }

            /// Pushes `vec` onto the collection
            pub fn push(self: &mut Self, vec: &Vector<'_>) {
                self.0.push(&vec.0[..]);
            }

            /// Returns the number of vectors in this collection.
            pub fn len(self: &Self) -> usize {
                self.0.len()
            }

            /// Returns a reference to the vector at `index`.
            pub fn get_element(self: &'a Self, index: usize) -> Vector<'a> {
                Vector(Cow::Borrowed(self.0.get(index)))
            }
        }

        impl<'a> FromIterator<Vector<'a>> for Vectors<'static> {
            fn from_iter<I: IntoIterator<Item = Vector<'a>>>(iter: I) -> Self {
                let mut vecs = Vectors::new();
                for vec in iter {
                    vecs.push(&vec);
                }

                vecs
            }
        }

        impl<'a> io::Writeable for Vectors<'a> {
            fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
                self.0.write(buffer)
            }
        }

        impl<'a> ElementContainer for Vectors<'a> {
            type Element = Vector<'static>;

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

        impl<'a> ExtendableElementContainer for Vectors<'a> {
            type InternalElement = Self::Element;

            fn push(self: &mut Self, element: Self::InternalElement) {
                self.push(&element)
            }
        }

        #[doc(hidden)]
        pub mod mmap {
            use super::*;
            use madvise::{AccessPattern, AdviseMemory};
            use memmap;

            pub struct MmapVectors {
                data: memmap::Mmap,
                dim: usize,
            }

            impl MmapVectors {
                pub fn new(filename: &str, dim: usize) -> Self {
                    let data = std::fs::File::open(filename).unwrap();
                    let data = unsafe { memmap::Mmap::map(&data).unwrap() };
                    data.advise_memory_access(AccessPattern::Random)
                        .expect("Error with madvise");

                    Self { data, dim }
                }

                pub fn load<'a>(self: &'a Self) -> Vectors<'a> {
                    Vectors::load(&self.data[..], self.dim)
                }
            }

            impl ElementContainer for MmapVectors {
                type Element = Vector<'static>;

                fn get(self: &Self, idx: usize) -> Self::Element {
                    self.load().get(idx)
                }

                fn len(self: &Self) -> usize {
                    self.load().len()
                }

                fn dist_to_element(
                    self: &Self,
                    idx: usize,
                    element: &Self::Element,
                ) -> NotNan<f32> {
                    self.load().dist_to_element(idx, element)
                }

                fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
                    self.load().dist(i, j)
                }

                fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
                    self.load().dists(idx, others)
                }
            }

            impl io::Writeable for MmapVectors {
                fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
                    self.load().write(buffer)
                }
            }
        }
    };
}
