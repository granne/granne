use super::{open_random_access_mmap, PyGranneBuilder, SaveIndex, WordDict};
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

impl PyGranneBuilder for granne::GranneBuilder<granne::angular::mmap::MmapVectors> {}
impl PyGranneBuilder for granne::GranneBuilder<granne::angular_int::mmap::MmapVectors> {}

pub struct WordEmbeddingsBuilder {
    builder: granne::GranneBuilder<granne::embeddings::mmap::MmapSumEmbeddings>,
    words: WordDict,
}

impl WordEmbeddingsBuilder {
    pub fn new(config: granne::BuildConfig, elements: &str, embeddings: &str, words: &str) -> Self {
        let words = WordDict::new(words);

        let dim = {
            let embeddings = open_random_access_mmap(embeddings);
            assert_eq!(
                0,
                embeddings.len() % (std::mem::size_of::<f32>() * words.len())
            );
            embeddings.len() / (std::mem::size_of::<f32>() * words.len())
        };

        let builder = granne::GranneBuilder::new(
            config,
            granne::embeddings::mmap::MmapSumEmbeddings::new(elements, embeddings, dim),
        );

        Self { builder, words }
    }
}

impl granne::Builder for WordEmbeddingsBuilder {
    fn build(self: &mut Self) {
        self.builder.build();
    }

    fn build_partial(self: &mut Self, num_elements: usize) {
        self.builder.build_partial(num_elements);
    }

    fn indexed_elements(self: &Self) -> usize {
        self.builder.indexed_elements()
    }

    fn write_index<B: std::io::Write + std::io::Seek>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<()> {
        self.builder.write_index(buffer)
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

impl PyGranneBuilder for WordEmbeddingsBuilder {}

macro_rules! impl_index {
    ($($type:ty),+) => {
        $(impl granne::Index for $type {
            fn len(self: &Self) -> usize {
                self.builder.get_index().len()
            }

            fn get_neighbors(self: &Self, idx: usize, layer: usize) -> Vec<usize> {
                self.builder.get_index().get_neighbors(idx, layer)
            }

            fn num_layers(self: &Self) -> usize {
                self.builder.get_index().num_layers()
            }

            fn layer_len(self: &Self, layer: usize) -> usize {
                self.builder.get_index().layer_len(layer)
            }
        })+
    }
}

impl_index!(WordEmbeddingsBuilder);
