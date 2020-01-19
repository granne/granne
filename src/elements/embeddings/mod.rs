//! A module for elements that can be embedded into vector spaces.

use ordered_float::NotNan;

use super::{angular, angular_int, Dist, ElementContainer, ExtendableElementContainer};
use crate::io;
use crate::slice_vector::VariableWidthSliceVector;
use crate::{math, FiveByteInt, ThreeByteInt};
use std::io::{Result, Write};

pub mod parsing;
pub mod reorder;

type Embeddings<'a> = angular::Vectors<'a>; // replace with something else??

type EmbeddingId = ThreeByteInt;
type ElementOffset = FiveByteInt;

type Elements<'a> = VariableWidthSliceVector<'a, EmbeddingId, ElementOffset>;

/// A data structure containing elements that can be embedded into a vector space.
/// `SumEmbeddings` consists of `elements` and `embeddings`. The vector for each
/// element is created by summing a subset of the vectors from `embeddings`.
///
/// # Example - text/sentence vectors based on word vectors:
///
/// `embeddings` contains a vector for any valid words. Each element is a text
/// snippet/sentence where each word has a corresponding vector in `embeddings`.
/// An element in `elements` is represented by the indices of its words in
/// `embeddings`, e.g. if
///
/// `embeddings = [v_{hello}, v_{rust}, v_{world}, ...]`,
///
/// and the element of interest is `hello world`, then its representation in `elements` would be `[0, 2]`.
///
#[derive(Clone)]
pub struct SumEmbeddings<'a> {
    embeddings: Embeddings<'a>,
    elements: Elements<'a>,
}

impl<'a> SumEmbeddings<'a> {
    /// Constructs `SumEmbeddings` with no elements.
    pub fn new(embeddings: Embeddings<'a>) -> Self {
        Self {
            embeddings,
            elements: Elements::new(),
        }
    }

    /// Loads `SumEmbeddings` from a buffer `elements`.
    pub fn load(embeddings: Embeddings<'a>, elements: &'a [u8]) -> Self {
        Self {
            embeddings,
            elements: Elements::load(elements),
        }
    }

    /// Constructs `SumEmbeddings` with `elements`.
    pub(crate) fn from_parts(embeddings: Embeddings<'a>, elements: Elements<'a>) -> Self {
        Self {
            embeddings,
            elements,
        }
    }

    /// Borrows the data
    pub fn borrow(self: &'a Self) -> SumEmbeddings<'a> {
        Self {
            embeddings: self.embeddings.borrow(),
            elements: self.elements.borrow(),
        }
    }

    /// Inserts a new element into the collection at the end.
    pub fn push<Element: AsRef<[usize]>>(self: &mut Self, element: Element) {
        let data: Vec<EmbeddingId> = element.as_ref().iter().map(|&id| id.into()).collect();
        self.elements.push(&data);
    }

    /// Returns the embedding ids for the element at `element_idx`.
    pub fn get_terms(self: &Self, element_idx: usize) -> Vec<usize> {
        self.elements
            .get(element_idx)
            .iter()
            .map(|&id| id.into())
            .collect()
    }

    /// Gets the (non-normalized) embedding for the element at `element_idx`.
    pub fn get_embedding(self: &Self, element_idx: usize) -> Vec<f32> {
        self.get_embedding_internal(self.elements.get(element_idx))
    }

    /// Creates a new (non-normalized) embedding for an element consisting of `embedding_ids`.
    pub fn create_embedding(self: &Self, embedding_ids: &[usize]) -> Vec<f32> {
        self.get_embedding_internal(embedding_ids)
    }

    /// Computes a raw (non-normalized) embedding for `embedding_ids`.
    /// Generic over `Id` in order to handle both &[usize] and &[EmbeddingId].
    fn get_embedding_internal<Id: Copy + Into<usize>>(
        self: &Self,
        embedding_ids: &[Id],
    ) -> Vec<f32> {
        if embedding_ids.is_empty() {
            if self.embeddings.len() > 0 {
                return vec![0.0f32; self.embeddings.get_element(0).len()];
            } else {
                return Vec::new();
            }
        }

        let w: usize = embedding_ids[0].into();
        let mut data: Vec<f32> = self.embeddings.get_element(w).to_vec();

        for w in embedding_ids.iter().skip(1).map(|&id| id.into()) {
            let embedding = self.embeddings.get_element(w);

            math::sum_into_f32(&mut data, embedding.as_slice());
        }

        data
    }

    /// Returns the number of elements in this collection.
    pub fn len(self: &Self) -> usize {
        self.elements.len()
    }

    /// Returns the number of embeddings in this collection.
    pub fn num_embeddings(self: &Self) -> usize {
        self.embeddings.len()
    }
}

impl<'a> ElementContainer for SumEmbeddings<'a> {
    type Element = angular::Vector<'static>;

    fn get(self: &Self, idx: usize) -> Self::Element {
        self.get_embedding(idx).into()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }

    fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
        self.get(idx).dist(element)
    }
}

impl<'a> ExtendableElementContainer for SumEmbeddings<'a> {
    type InternalElement = Vec<usize>;

    fn push(self: &mut Self, element: Self::InternalElement) {
        self.push(element)
    }
}

impl<'a> io::Writeable for SumEmbeddings<'a> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        self.elements.write(buffer)
    }
}

#[doc(hidden)]
pub mod mmap {
    use super::*;
    use madvise::{AccessPattern, AdviseMemory};
    use memmap;

    pub struct MmapSumEmbeddings {
        elements: memmap::Mmap,
        embeddings: memmap::Mmap,
        dim: usize,
    }

    impl MmapSumEmbeddings {
        pub fn new(elements: &str, embeddings: &str, dim: usize) -> Self {
            let elements = std::fs::File::open(elements).unwrap();
            let elements = unsafe { memmap::Mmap::map(&elements).unwrap() };
            elements
                .advise_memory_access(AccessPattern::Random)
                .expect("Error with madvise");

            let embeddings = std::fs::File::open(embeddings).unwrap();
            let embeddings = unsafe { memmap::Mmap::map(&embeddings).unwrap() };
            embeddings
                .advise_memory_access(AccessPattern::Random)
                .expect("Error with madvise");

            Self {
                elements,
                embeddings,
                dim,
            }
        }

        pub fn load<'a>(self: &'a Self) -> SumEmbeddings<'a> {
            let embeddings = Embeddings::load(&self.embeddings[..], self.dim);
            SumEmbeddings::load(embeddings, &self.elements[..])
        }
    }

    impl ElementContainer for MmapSumEmbeddings {
        type Element = angular::Vector<'static>;

        fn get(self: &Self, idx: usize) -> Self::Element {
            self.load().get(idx)
        }

        fn len(self: &Self) -> usize {
            self.load().len()
        }

        fn dist_to_element(self: &Self, idx: usize, element: &Self::Element) -> NotNan<f32> {
            self.load().dist_to_element(idx, element)
        }

        fn dist(self: &Self, i: usize, j: usize) -> NotNan<f32> {
            self.load().dist(i, j)
        }

        fn dists(self: &Self, idx: usize, others: &[usize]) -> Vec<NotNan<f32>> {
            self.load().dists(idx, others)
        }
    }

    impl io::Writeable for MmapSumEmbeddings {
        fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
            self.load().write(buffer)
        }
    }
}

impl<'a> super::Permutable for SumEmbeddings<'a> {
    fn permute(self: &mut Self, permutation: &[usize]) {
        use pbr::ProgressBar;
        use rayon::prelude::*;

        assert_eq!(self.len(), permutation.len());

        let show_progress = true;
        let mut progress_bar = if show_progress {
            Some(ProgressBar::new(self.len() as u64))
        } else {
            None
        };

        let chunk_size = std::cmp::max(10_000, self.len() / 400);
        let chunks: Vec<_> = permutation
            .par_chunks(chunk_size)
            .map(|c| {
                let mut elements = Elements::new();
                for &id in c {
                    elements.push(self.elements.get(id));
                }
                elements
            })
            .collect();

        let mut new_elements = Elements::new();
        for chunk in chunks {
            new_elements.extend_from_slice_vector(&chunk);

            if let Some(ref mut progress_bar) = progress_bar {
                progress_bar.set(new_elements.len() as u64);
            }
        }

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.finish_println("");
        }

        self.elements = new_elements;
    }
}
