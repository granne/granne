use ordered_float::NotNan;

use super::{AngularVector, AngularVectors, Dist, ElementContainer, ExtendableElementContainer};
use crate::io;
use crate::slice_vector::VariableWidthSliceVector;
use crate::{FiveByteInt, ThreeByteInt};
use std::io::{Result, Write};

pub mod parsing;
pub mod reorder;

type Embeddings<'a> = AngularVectors<'a>;

type EmbeddingId = ThreeByteInt;
type ElementOffset = FiveByteInt;

type Elements<'a> = VariableWidthSliceVector<'a, EmbeddingId, ElementOffset>;

/// A data structure representing ... where each element vector is created by summing a number of vectors
/// from `embeddings`.
///
/// # Example - text/sentence vectors based on word vectors:
///
/// `embeddings` contains vectors for all valid words. Each element is a text snippet/sentence
/// where each word has a corresponding vector in `embeddings`. An element in `elements` is represented
/// by the indices of its words in `embeddings`, e.g. if
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

    /// Constructs `SumEmbeddings` with `elements`.
    pub fn from_parts(embeddings: Embeddings<'a>, elements: Elements<'a>) -> Self {
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

        // speed up
        for w in embedding_ids.iter().skip(1).map(|&id| id.into()) {
            let embedding = self.embeddings.get_element(w);

            assert_eq!(data.len(), embedding.len());

            let embedding = embedding.as_slice();
            for i in 0..data.len() {
                data[i] += embedding[i];
            }
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
    type Element = AngularVector<'static>;

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
