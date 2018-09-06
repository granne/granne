use cpython::{PyObject, PyResult, Python};
use granne::query_embeddings::{QueryEmbeddings, DIM};
use granne::{At, Dense};
use granne;
use madvise::{AccessPattern, AdviseMemory};
use memmap;
use serde_json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use super::DEFAULT_NUM_NEIGHBORS;
use super::DEFAULT_MAX_SEARCH;

type IndexType<'a> = granne::Hnsw<'a, QueryEmbeddings<'a>, granne::AngularVector<[f32; DIM]>>;
type BuilderType = granne::HnswBuilder<'static, MmapQueryEmbeddings, granne::AngularVector<[f32; DIM]>>;

pub struct WordDict {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>
}

impl WordDict {
    pub fn new(path: &str) -> Self {
        let word_file = File::open(&path).unwrap();
        let word_file = BufReader::new(word_file);
        let words: Vec<String> = word_file
            .lines()
            .map(|w| {
                let w = w.unwrap();
                serde_json::from_str::<String>(&w).unwrap()
            })
            .collect();

        let word_to_id = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i))
            .collect();

        Self {
            word_to_id: word_to_id,
            id_to_word: words
        }
    }

    pub fn get_query(self: &Self, ids: &[usize]) -> String {
        if ids.is_empty() {
            return String::new()
        }

        let mut query = self.id_to_word[ids[0]].clone();

        for word in ids[1..].iter().map(|&id| self.id_to_word[id].as_str()) {
            query.push(' ');
            query.push_str(word);
        }

        query
    }

    pub fn get_word_ids(self: &Self, query: &str) -> Vec<usize> {
        query.split_whitespace().filter_map(|w| self.word_to_id.get(w).cloned()).collect()
    }
}

py_class!(pub class QueryHnsw |py| {
    data word_embeddings: memmap::Mmap;
    data index: memmap::Mmap;
    data elements: memmap::Mmap;
    data word_dict: Option<WordDict>;

    def __new__(_cls,
                word_embeddings_path: &str,
                index_path: &str,
                elements_path: &str,
                words_path: Option<String> = None) -> PyResult<QueryHnsw> {

        let index = File::open(index_path).unwrap();
        let index = unsafe { memmap::Mmap::map(&index).unwrap() };

        index.advise_memory_access(AccessPattern::Random).expect("Error with madvise");

        let word_embeddings = File::open(word_embeddings_path).unwrap();
        let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

        let elements = File::open(elements_path).unwrap();
        let elements = unsafe { memmap::Mmap::map(&elements).unwrap() };

        // sanity check / fail early
        {
            let elements = QueryEmbeddings::load(&word_embeddings, &elements);
            let _index = IndexType::load(&index, &elements);
        }

        let word_dict = match words_path {
            Some(words_path) => Some(WordDict::new(&words_path)),
            None => None
        };

        QueryHnsw::create_instance(py, word_embeddings, index, elements, word_dict)
    }

    def search(&self,
               element: Vec<f32>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let elements = QueryEmbeddings::load(&self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.search(
            &element.into_iter().collect(), num_elements, max_search))
    }

    def search_query(&self,
                     query: &str,
                     num_elements: usize = DEFAULT_NUM_NEIGHBORS,
                     max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, String, f32)>>
    {
        let word_dict = self.word_dict(py).as_ref().unwrap();
        let elements = QueryEmbeddings::load(&self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        let element = elements.get_embedding_for_query(&word_dict.get_word_ids(query));

        let result = index.search(&element, num_elements, max_search);

        Ok(result.into_iter().map(|(id, dist)| (id, word_dict.get_query(&elements.get_words(id)), dist)).collect())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let elements = QueryEmbeddings::load(&self.word_embeddings(py), &self.elements(py));
        Ok(elements.get_embedding(idx).0.to_vec())
    }

    def get_query(&self, idx: usize) -> PyResult<String> {
        let word_dict = self.word_dict(py).as_ref().unwrap();
        let elements = QueryEmbeddings::load(&self.word_embeddings(py), &self.elements(py));

        Ok(word_dict.get_query(&elements.get_words(idx)))
    }

    def __len__(&self) -> PyResult<usize> {
        let elements = QueryEmbeddings::load(&self.word_embeddings(py), &self.elements(py));
        Ok(elements.len())
    }

});


/// A wrapper around granne::QueryEmbeddings to allow building indexes from python
#[derive(Clone)]
pub struct MmapQueryEmbeddings {
    word_embeddings: Arc<memmap::Mmap>,
    queries: Arc<memmap::Mmap>
}

impl MmapQueryEmbeddings {
    pub fn new(word_embeddings: memmap::Mmap, queries: memmap::Mmap) -> Self {
        Self {
            word_embeddings: Arc::new(word_embeddings),
            queries: Arc::new(queries)
        }
    }

    #[inline(always)]
    pub fn load<'a>(self: &'a Self) -> QueryEmbeddings<'a> {
        QueryEmbeddings::load(&self.word_embeddings[..], &self.queries[..])
    }
}


impl At for MmapQueryEmbeddings {
    type Output = granne::AngularVector<[f32; DIM]>;

    fn at(self: &Self, index: usize) -> Self::Output {
        self.load().at(index)
    }

    fn len(self: &Self) -> usize {
        self.load().len()
    }
}


pub fn py_parse_queries_and_save_to_disk(py: Python, queries_path: String, words_path: String, output_path: String, show_progress: bool) -> PyResult<PyObject>{
    granne::query_embeddings::parsing::parse_queries_and_save_to_disk(
        &Path::new(&queries_path),
        &Path::new(&words_path),
        &Path::new(&output_path),
        show_progress
    );

    Ok(py.None())
}

pub fn py_compute_query_vectors_and_save_to_disk(py: Python, queries_path: String, word_embeddings_path: String, output_path: String, show_progress: bool) -> PyResult<PyObject>{
    granne::query_embeddings::parsing::compute_query_vectors_and_save_to_disk(
        &Path::new(&queries_path),
        &Path::new(&word_embeddings_path),
        &Path::new(&output_path),
        show_progress
    );

    Ok(py.None())
}


py_class!(pub class QueryHnswBuilder |py| {
    data builder: RefCell<BuilderType>;

    @classmethod
    def with_mmapped_elements(_cls,
                              num_layers: usize,
                              word_embeddings_path: &str,
                              queries_path: &str,
                              index_path: Option<String> = None,
                              max_search: usize = DEFAULT_MAX_SEARCH,
                              show_progress: bool = true,
                              words_path: Option<String> = None) -> PyResult<QueryHnswBuilder>
    {
        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let word_embeddings = File::open(word_embeddings_path).unwrap();
        let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

        let queries = File::open(queries_path).unwrap();
        let queries = unsafe { memmap::Mmap::map(&queries).unwrap() };

        let elements = MmapQueryEmbeddings::new(word_embeddings, queries);

        // sanity check / fail early
        {
            let _ = elements.len();
        }


        let builder = if let Some(index_path) = index_path {
            let index_file = File::open(index_path).expect("Could not open index file");
            let mut index_file = BufReader::new(index_file);

            BuilderType::read_index_with_owned_elements(config, &mut index_file, elements).expect("Could not read existing index")

        } else {
            BuilderType::with_owned_elements(config, elements)
        };

        QueryHnswBuilder::create_instance(py, RefCell::new(builder))
    }

    def __len__(&self) -> PyResult<usize> {
        let builder = self.builder(py).borrow();

        Ok(builder.len())
    }

    def indexed_elements(&self) -> PyResult<usize> {
        let builder = self.builder(py).borrow();

        Ok(builder.indexed_elements())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let builder = self.builder(py).borrow();
        let index = builder.get_index();

        Ok(index.get_element(idx).as_slice().to_vec())
    }


    def build_index(&self, num_elements: usize = <usize>::max_value()) -> PyResult<PyObject>{
        let mut builder = self.builder(py).borrow_mut();
        if num_elements == <usize>::max_value() {
            builder.build_index();
        } else {
            builder.build_index_part(num_elements);
        }

        Ok(py.None())
    }

    def save_index(&self, path: &str) -> PyResult<PyObject> {
        let builder = self.builder(py).borrow();
        builder.save_index_to_disk(path).expect("Could not save index to disk");

        return Ok(py.None())
    }

    def search(&self, element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let builder = self.builder(py).borrow();
        let index = builder.get_index();

        return Ok(index.search(&element.into_iter().collect(), num_neighbors, max_search))
    }
});
