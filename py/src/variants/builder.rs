use super::WordDict;
use crate::{AsBuilder, AsIndex, PyGranneBuilder, SaveIndex};
use cpython::{FromPyObject, PyObject, PyResult, Python};
use granne;

impl<'a> PyGranneBuilder for granne::GranneBuilder<granne::angular::Vectors<'a>> {
    fn push(self: &mut Self, py: Python, element: &PyObject) -> PyResult<PyObject> {
        let element = granne::angular::Vector::from(Vec::extract(py, element)?);
        self.push(element);

        Ok(py.None())
    }
}

impl<'a> PyGranneBuilder for granne::GranneBuilder<granne::angular_int::Vectors<'a>> {
    fn push(self: &mut Self, py: Python, element: &PyObject) -> PyResult<PyObject> {
        let element = granne::angular_int::Vector::from(Vec::extract(py, element)?);
        self.push(element);

        Ok(py.None())
    }
}

pub struct WordEmbeddingsBuilder {
    builder: granne::GranneBuilder<granne::embeddings::SumEmbeddings<'static>>,
    words: WordDict,
}

impl WordEmbeddingsBuilder {
    pub fn new(
        config: granne::BuildConfig,
        elements: Option<&std::fs::File>,
        embeddings: &std::fs::File,
        words: &str,
        index: Option<&std::fs::File>,
    ) -> Self {
        let words = WordDict::new(words);

        let builder = if let Some(index) = index {
            granne::GranneBuilder::from_file(config, index, unsafe {
                granne::embeddings::SumEmbeddings::from_files(embeddings, elements).expect("Could not load elements.")
            })
            .expect("Could not read index.")
        } else {
            granne::GranneBuilder::new(config, unsafe {
                granne::embeddings::SumEmbeddings::from_files(embeddings, elements).expect("Could not load elements.")
            })
        };

        Self { builder, words }
    }
}

impl AsBuilder for WordEmbeddingsBuilder {
    fn as_builder(self: &Self) -> &dyn granne::Builder {
        &self.builder
    }

    fn as_mut_builder(self: &mut Self) -> &mut dyn granne::Builder {
        &mut self.builder
    }
}

impl AsIndex for WordEmbeddingsBuilder {
    fn as_index(self: &Self) -> &dyn granne::Index {
        &self.builder
    }

    fn as_mut_index(self: &mut Self) -> &mut dyn granne::Index {
        &mut self.builder
    }
}

impl SaveIndex for WordEmbeddingsBuilder {
    fn save_index(self: &Self, path: &str) -> std::io::Result<()> {
        self.builder.save_index(path)
    }

    fn save_elements(self: &Self, path: &str) -> std::io::Result<()> {
        self.builder.save_elements(path)
    }
}

impl PyGranneBuilder for WordEmbeddingsBuilder {
    fn push(self: &mut Self, py: Python, element: &PyObject) -> PyResult<PyObject> {
        let words = String::extract(py, element)?;
        let ids = self.words.get_word_ids(&words);

        self.builder.push(ids);

        Ok(py.None())
    }
}
