macro_rules! dense_vector {
    ($scalar_type:ty) => {
        /// A vector element.
        #[derive(Clone)]
        pub struct Vector<'a>(pub Cow<'a, [$scalar_type]>);

        impl<'a> Vector<'a> {
            /// Returns the number of elements in this `Vector`.
            pub fn len(self: &Self) -> usize {
                self.0.len()
            }

            /// Clones the underlying data if not already owned.
            pub fn into_owned(self: Self) -> Vector<'static> {
                Vector(self.0.into_owned().into())
            }

            /// Converts this `Vector` into a `Vec`.
            pub fn into_vec(self: Self) -> Vec<$scalar_type> {
                self.0.into_owned()
            }

            /// Returns a reference to the underlying slice.
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

        /// A collection of `Vector`s.
        #[derive(Clone)]
        pub struct Vectors<'a>(FixedWidthSliceVector<'a, $scalar_type>);

        impl<'a> Vectors<'a> {
            /// Creates a new collection vector. The dimension will be set once the first vector is
            /// pushed into the collection.
            pub fn new() -> Self {
                Self(FixedWidthSliceVector::new())
            }

            /// Loads a collection of vectors from a `u8` buffer.
            /// `buffer` needs to contain data in a compatible format (e.g. written with
            /// `Vectors::write`).
            pub fn from_bytes(buffer: &'a [u8]) -> Self {
                Self(FixedWidthSliceVector::from_bytes(buffer))
            }

            /// Loads a memory-mapped a collection of vectors from a file.
            ///
            /// ## Safety
            ///
            /// This is unsafe because the underlying file can be modified, which would result in
            /// undefined behavior. The caller needs to guarantee that the file is not modified
            /// while being memory-mapped.
            pub unsafe fn from_file(file: &std::fs::File) -> std::io::Result<Self> {
                Ok(Self(FixedWidthSliceVector::from_file(file)?))
            }

            /// Creates a collection of vectors with dimension `dim` from a slice.
            ///
            /// `dim` needs to be non-zero and divide the length of `vec`.
            pub fn from_slice(slice: &'a [$scalar_type], dim: usize) -> Self {
                Self(FixedWidthSliceVector::with_data(slice, dim))
            }

            /// Creates a collection of vectors with dimension `dim` from a `Vec`.
            ///
            /// `dim` needs to be non-zero and divide the length of `vec`.
            pub fn from_vec(vec: Vec<$scalar_type>, dim: usize) -> Self {
                Self(FixedWidthSliceVector::with_data(vec, dim))
            }

            /// Borrows the data.
            pub fn borrow(self: &'a Self) -> Vectors<'a> {
                Self(self.0.borrow())
            }

            /// Clones the underlying data if not already owned.
            pub fn into_owned(self: Self) -> Vectors<'static> {
                Vectors(self.0.into_owned())
            }

            /// Extends `Vectors` with the elements from `vec`.
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

            /// Returns the dimension of each vector in this collection.
            pub fn dim(self: &Self) -> usize {
                self.0.width()
            }

            /// Returns a reference to the vector at `index`.
            pub fn get_element(self: &'a Self, index: usize) -> Vector<'a> {
                Vector(Cow::Borrowed(self.0.get(index)))
            }

            /// Returns a reference to the underlying slice.
            pub fn as_slice(self: &Self) -> &[$scalar_type] {
                self.0.as_slice()
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
            /// Writes `Vectors` to a `buffer`.
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

        use crate::Permutable;

        impl<'a> Permutable for Vectors<'a> {
            fn permute(self: &mut Self, permutation: &[usize]) {
                self.0.permute(permutation);
            }
        }
    };
}
