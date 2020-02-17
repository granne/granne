//! A module for elements that can be embedded into vector spaces.

use ordered_float::NotNan;

use super::{angular, Dist, ElementContainer, ExtendableElementContainer};
use crate::io;
use crate::slice_vector::{FixedWidthSliceVector, VariableWidthSliceVector};
use crate::{math, FiveByteInt, ThreeByteInt};
use std::io::{Result, Write};

#[doc(hidden)]
pub mod parsing;

mod reorder;

pub use reorder::compute_keys_for_reordering;

type Embeddings<'a> = FixedWidthSliceVector<'a, f32>;

type EmbeddingId = ThreeByteInt;
type ElementOffset = FiveByteInt;

type Elements<'a> = VariableWidthSliceVector<'a, EmbeddingId, ElementOffset>;

/// A data structure containing elements that can be embedded into a vector space.
/// `SumEmbeddings` consists of `embeddings` and `elements`. The vector for each
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
/// and the element of interest is `hello world`, then its representation in `elements` would be
/// `[0, 2]`.
#[derive(Clone, Default)]
pub struct SumEmbeddings<'a> {
    embeddings: Embeddings<'a>,
    elements: Elements<'a>,
}

impl<'a> SumEmbeddings<'a> {
    /// Constructs empty `SumEmbeddings` with no embeddings nor elements.
    pub fn new() -> Self {
        Self {
            embeddings: Embeddings::new(),
            elements: Elements::new(),
        }
    }

    /// Loads `SumEmbeddings` from a buffer `elements`.
    pub fn from_bytes(embeddings: Embeddings<'a>, elements: &'a [u8]) -> Self {
        Self {
            embeddings,
            elements: Elements::from_bytes(elements),
        }
    }

    /// Constructs `SumEmbeddings` with `elements`.
    pub(crate) fn from_parts(embeddings: Embeddings<'a>, elements: Elements<'a>) -> Self {
        Self { embeddings, elements }
    }

    /// Loads a memory-mapped `SumEmbeddings` from `embeddings` (and optionally `elements`).
    ///
    /// ## Safety
    ///
    /// This is unsafe because the underlying file can be modified, which would result in undefined
    /// behavior. The caller needs to guarantee that the file is not modified while being
    /// memory-mapped.
    pub unsafe fn from_files(embeddings: &std::fs::File, elements: Option<&std::fs::File>) -> std::io::Result<Self> {
        let elements = if let Some(elements) = elements {
            Elements::from_file(elements)?
        } else {
            Elements::new()
        };

        Ok(Self {
            embeddings: Embeddings::from_file(embeddings)?,
            elements,
        })
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

    /// Inserts a new element into the collection at the end.
    pub fn push_embedding(self: &mut Self, embedding: &[f32]) {
        self.embeddings.push(embedding)
    }

    /// Returns the embedding ids for the element at `element_idx`.
    pub fn get_terms(self: &Self, element_idx: usize) -> Vec<usize> {
        self.elements.get(element_idx).iter().map(|&id| id.into()).collect()
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
    fn get_embedding_internal<Id: Copy + Into<usize>>(self: &Self, embedding_ids: &[Id]) -> Vec<f32> {
        if embedding_ids.is_empty() {
            if self.embeddings.len() > 0 {
                return vec![0.0f32; self.embeddings.width()];
            } else {
                return Vec::new();
            }
        }

        let w: usize = embedding_ids[0].into();
        let mut data: Vec<f32> = self.embeddings.get(w).to_vec();

        for w in embedding_ids.iter().skip(1).map(|&id| id.into()) {
            let embedding = self.embeddings.get(w);

            math::sum_into_f32(&mut data, embedding);
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

    /// Writes the embeddings to `buffer`.
    pub fn write_embeddings<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        self.embeddings.write(buffer)
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

impl<'a> super::Permutable for SumEmbeddings<'a> {
    fn permute(self: &mut Self, permutation: &[usize]) {
        use rayon::prelude::*;

        assert_eq!(self.len(), permutation.len());

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
        }

        self.elements = new_elements;
    }
}
