use cpython::{FromPyObject, PyObject, PyResult, Python, PythonObject, ToPyObject};

use super::{open_random_access_mmap, PyGranne, WordDict};
use granne;

pub struct AngularGranne {
    index: memmap::Mmap,
    elements: memmap::Mmap,
    dim: usize,
}

impl AngularGranne {
    pub fn new(index: &str, elements: &str, dim: usize) -> Self {
        Self {
            index: open_random_access_mmap(index),
            elements: open_random_access_mmap(elements),
            dim,
        }
    }

    fn load_index(self: &Self) -> granne::Granne<granne::angular::Vectors> {
        let elements = granne::angular::Vectors::load(&self.elements[..], self.dim);
        granne::Granne::load(&self.index[..], elements)
    }
}

impl PyGranne for AngularGranne {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let element = granne::angular::Vector::from(Vec::extract(py, element)?);
        Ok(self.load_index().search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.load_index()
            .get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }
}

pub struct AngularIntGranne {
    index: memmap::Mmap,
    elements: memmap::Mmap,
    dim: usize,
}

impl AngularIntGranne {
    pub fn new(index: &str, elements: &str, dim: usize) -> Self {
        Self {
            index: open_random_access_mmap(index),
            elements: open_random_access_mmap(elements),
            dim,
        }
    }

    fn load_index(self: &Self) -> granne::Granne<granne::angular_int::Vectors> {
        let elements = granne::angular_int::Vectors::load(&self.elements[..], self.dim);
        granne::Granne::load(&self.index[..], elements)
    }
}

impl PyGranne for AngularIntGranne {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let element = granne::angular_int::Vector::from(Vec::extract(py, element)?);
        Ok(self.load_index().search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.load_index()
            .get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }
}

pub struct WordEmbeddingsGranne {
    index: memmap::Mmap,
    elements: memmap::Mmap,
    embeddings: memmap::Mmap,
    dim: usize,
    words: WordDict,
}

impl WordEmbeddingsGranne {
    pub fn new(index: &str, elements: &str, embeddings: &str, words: &str) -> Self {
        let words = WordDict::new(words);

        let embeddings = open_random_access_mmap(embeddings);

        assert_eq!(
            0,
            embeddings.len() % (std::mem::size_of::<f32>() * words.len())
        );
        let dim = embeddings.len() / (std::mem::size_of::<f32>() * words.len());

        Self {
            index: open_random_access_mmap(index),
            elements: open_random_access_mmap(elements),
            embeddings,
            dim,
            words,
        }
    }

    fn load_embeddings(self: &Self) -> granne::embeddings::SumEmbeddings {
        let embeddings = granne::angular::Vectors::load(&self.embeddings[..], self.dim);
        let elements = granne::embeddings::SumEmbeddings::load(embeddings, &self.elements[..]);

        elements
    }

    fn load_index(self: &Self) -> granne::Granne<granne::embeddings::SumEmbeddings> {
        granne::Granne::load(&self.index[..], self.load_embeddings())
    }
}

impl PyGranne for WordEmbeddingsGranne {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let element =
            granne::angular::Vector::from(if let Ok(element) = Vec::<f32>::extract(py, element) {
                element
            } else {
                let words = String::extract(py, element)?;
                let ids = self.words.get_word_ids(&words);

                self.load_embeddings().create_embedding(&ids)
            });

        Ok(self.load_index().search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.load_index()
            .get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }

    fn get_internal_element(self: &Self, py: Python, idx: usize) -> PyObject {
        let idxs = self.load_embeddings().get_terms(idx);
        let words = self.words.get_words(&idxs);

        words.into_py_object(py).into_object()
    }
}

macro_rules! impl_index {
    ($($type:ty),+) => {
        $(impl granne::Index for $type {
            fn len(self: &Self) -> usize {
                self.load_index().len()
            }

            fn get_neighbors(self: &Self, idx: usize, layer: usize) -> Vec<usize> {
                self.load_index().get_neighbors(idx, layer)
            }

            fn num_layers(self: &Self) -> usize {
                self.load_index().num_layers()
            }

            fn layer_len(self: &Self, layer: usize) -> usize {
                self.load_index().layer_len(layer)
            }
        })+
    }
}

impl_index!(AngularGranne, AngularIntGranne, WordEmbeddingsGranne);
