use cpython::{FromPyObject, PyObject, PyResult, Python, PythonObject, ToPyObject};

use super::WordDict;
use crate::{AsIndex, PyGranne};
use granne;

impl<'a> PyGranne for granne::Granne<'a, granne::angular::Vectors<'a>> {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let element = granne::angular::Vector::from(Vec::extract(py, element)?);
        Ok(self.search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }
}

impl<'a> PyGranne for granne::Granne<'a, granne::angular_int::Vectors<'a>> {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let element = granne::angular_int::Vector::from(Vec::extract(py, element)?);
        Ok(self.search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }
}

pub struct WordEmbeddingsGranne {
    index: granne::Granne<'static, granne::embeddings::SumEmbeddings<'static>>,
    words: WordDict,
}

impl WordEmbeddingsGranne {
    pub fn new(
        index: &std::fs::File,
        elements: &std::fs::File,
        embeddings: &std::fs::File,
        words: &str,
    ) -> Self {
        let words = WordDict::new(words);

        let elements = unsafe {
            granne::embeddings::SumEmbeddings::from_files(embeddings, Some(elements))
                .expect("Could not open embeddings/elements")
        };

        let index =
            unsafe { granne::Granne::from_file(index, elements).expect("Could not load index.") };

        Self { index, words }
    }
}

impl AsIndex for WordEmbeddingsGranne {
    fn as_index(self: &Self) -> &dyn granne::Index {
        &self.index
    }

    fn as_mut_index(self: &mut Self) -> &mut dyn granne::Index {
        &mut self.index
    }
}

impl crate::Reorder for WordEmbeddingsGranne {
    fn reorder(self: &mut Self, show_progress: bool) -> Vec<usize> {
        self.index.reorder(show_progress)
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
        let data: Vec<f32> = if let Ok(data) = Vec::<f32>::extract(py, element) {
            data
        } else {
            let words = String::extract(py, element)?;
            let ids = self.words.get_word_ids(&words);

            self.index.get_elements().create_embedding(&ids)
        };

        let element = granne::angular::Vector::from(data);

        Ok(self.index.search(&element, max_search, num_elements))
    }

    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.index
            .get_element(idx)
            .to_vec()
            .into_py_object(py)
            .into_object()
    }

    fn get_internal_element(self: &Self, py: Python, idx: usize) -> PyObject {
        let idxs = self.index.get_elements().get_terms(idx);
        let words = self.words.get_words(&idxs);

        words.into_py_object(py).into_object()
    }
}
