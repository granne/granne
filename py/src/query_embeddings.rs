use cpython::{PyObject, PyResult, Python};
use granne;
use granne::{At, ComparableTo, Dense};
use madvise::{AccessPattern, AdviseMemory};
use memmap;
use rayon::prelude::*;
use serde_json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use super::{DEFAULT_MAX_SEARCH, DEFAULT_NUM_NEIGHBORS, DTYPE};

type IndexType<'a> = granne::Hnsw<'a, granne::QueryEmbeddings<'a>, granne::AngularVector<'a>>;
type BuilderType = granne::HnswBuilder<'static, MmapQueryEmbeddings, granne::AngularVector<'static>>;

pub struct WordDict {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
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

        let word_to_id = words.iter().enumerate().map(|(i, w)| (w.to_string(), i)).collect();

        Self {
            word_to_id: word_to_id,
            id_to_word: words,
        }
    }

    pub fn len(self: &Self) -> usize {
        self.id_to_word.len()
    }

    pub fn get_query(self: &Self, ids: &[usize]) -> String {
        if ids.is_empty() {
            return String::new();
        }

        let mut query = self.id_to_word[ids[0]].clone();

        for word in ids[1..].iter().map(|&id| self.id_to_word[id].as_str()) {
            query.push(' ');
            query.push_str(word);
        }

        query
    }

    pub fn get_word_ids(self: &Self, query: &str) -> Vec<usize> {
        query
            .split_whitespace()
            .filter_map(|w| self.word_to_id.get(w).cloned())
            .collect()
    }
}

py_class!(pub class Embeddings |py| {
    data word_embeddings: memmap::Mmap;
    data word_dict: WordDict;
    data dimension: usize;

    def __new__(_cls,
                word_embeddings_path: &str,
                words_path: &str) -> PyResult<Embeddings> {


        let word_dict = WordDict::new(&words_path);

        let word_embeddings = File::open(word_embeddings_path).unwrap();
        let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

        let dimension = word_embeddings.len() / (::std::mem::size_of::<f32>() * word_dict.len());

        Embeddings::create_instance(py, word_embeddings, word_dict, dimension)
    }

    def dim(&self) -> PyResult<usize> {
        Ok(*self.dimension(py))
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.word_dict(py).len())
    }

    def __getitem__(&self, idx: usize) -> PyResult<(String, Vec<f32>)> {
        let word_embeddings = granne::WordEmbeddings::load(*self.dimension(py), &self.word_embeddings(py)[..]);

        let word = self.word_dict(py).get_query(&[idx]);
        let embedding = word_embeddings.get_embedding(&[idx]).into();

        Ok((word, embedding))
    }

    def get(&self, q: &str) -> PyResult<Vec<f32>> {
        let word_embeddings = granne::WordEmbeddings::load(*self.dimension(py), &self.word_embeddings(py)[..]);
        let embeddings = granne::QueryEmbeddings::new(word_embeddings);

        Ok(embeddings.get_embedding_for_query(&self.word_dict(py).get_word_ids(q)).into())
    }

    def dist(&self, q0: &str, q1: &str) -> PyResult<f32> {
        Ok(self.dists(py, q0, vec![q1.to_string()]).unwrap()[0])
    }

    def dists(&self, query: &str, queries: Vec<String>) -> PyResult<Vec<f32>> {
        let word_embeddings = granne::WordEmbeddings::load(*self.dimension(py), &self.word_embeddings(py)[..]);
        let embeddings = granne::QueryEmbeddings::new(word_embeddings);

        let query = embeddings.get_embedding_for_query(&self.word_dict(py).get_word_ids(query));

        let distances = queries.into_iter().map(|q| query.dist(
            &embeddings.get_embedding_for_query(
                &self.word_dict(py).get_word_ids(&q)
            )).into_inner()
        ).collect();

        Ok(distances)
    }

    def dists_par(&self, query: &str, queries: Vec<String>) -> PyResult<Vec<f32>> {
        let word_embeddings = granne::WordEmbeddings::load(*self.dimension(py), &self.word_embeddings(py)[..]);
        let embeddings = granne::QueryEmbeddings::new(word_embeddings);

        let word_dict = self.word_dict(py);
        let query = embeddings.get_embedding_for_query(&word_dict.get_word_ids(query));

        let distances = queries.into_par_iter().map(|q| query.dist(
            &embeddings.get_embedding_for_query(
                &word_dict.get_word_ids(&q)
            )).into_inner()
        ).collect();

        Ok(distances)
    }

    def get_term_relevances(&self, query: &str) -> PyResult<Vec<f32>> {
        let word_embeddings = granne::WordEmbeddings::load(*self.dimension(py), &self.word_embeddings(py)[..]);

        let norm = |v: &[f32]| { v.iter().map(|x| x*x).sum::<f32>().sqrt() };

        Ok(self.word_dict(py).get_word_ids(query).into_iter().map(|id| norm(&word_embeddings.get_raw_embedding(&[id]))).collect())
    }
});

py_class!(pub class QueryHnsw |py| {
    data word_embeddings: memmap::Mmap;
    data index: memmap::Mmap;
    data elements: memmap::Mmap;
    data dimension: usize;
    data word_dict: Option<WordDict>;

    def __new__(_cls,
                word_embeddings_path: &str,
                index_path: &str,
                elements_path: &str,
                dimension: usize,
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
            let elements = granne::QueryEmbeddings::load(dimension, &word_embeddings, &elements);
            let _index = IndexType::load(&index, &elements);
        }

        let word_dict = match words_path {
            Some(words_path) => Some(WordDict::new(&words_path)),
            None => None
        };

        QueryHnsw::create_instance(py, word_embeddings, index, elements, dimension, word_dict)
    }

    def search(&self,
               element: Vec<f32>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.search(
            &element.into_iter().collect(), num_elements, max_search))
    }

    def search_batch(&self,
                     elements: Vec<Vec<f32>>,
                     num_elements: usize = DEFAULT_NUM_NEIGHBORS,
                     max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<Vec<(usize, f32)>>>
    {
        let _elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &_elements);

        Ok(elements
           .into_par_iter()
           .map(|element| index.search(&element.into_iter().collect(), num_elements, max_search))
           .collect()
        )
    }

    def search_query(&self,
                     query: &str,
                     num_elements: usize = DEFAULT_NUM_NEIGHBORS,
                     max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, String, f32)>>
    {
        let word_dict = self.word_dict(py).as_ref().unwrap();
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        let element = elements.get_embedding_for_query(&word_dict.get_word_ids(query));

        let result = index.search(&element, num_elements, max_search);

        Ok(result.into_iter().map(|(id, dist)| (id, word_dict.get_query(&elements.get_words(id)), dist)).collect())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        Ok(elements.get_embedding(idx).into())
    }

    def get_query(&self, idx: usize) -> PyResult<String> {
        let word_dict = self.word_dict(py).as_ref().unwrap();
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));

        Ok(word_dict.get_query(&elements.get_words(idx)))
    }

    def __len__(&self) -> PyResult<usize> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        Ok(elements.len())
    }

    def get_neighbors(&self, idx: usize, layer: usize) -> PyResult<Vec<usize>> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.get_neighbors(idx, layer))
    }

    def layer_len(&self, layer: usize) -> PyResult<usize> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.layer_len(layer))
    }

    def count_neighbors(&self, layer: usize) -> PyResult<usize> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.count_neighbors(layer))
    }

    def count_some_neighbors(&self, layer: usize, start: usize, stop: usize) -> PyResult<usize> {
        let elements = granne::QueryEmbeddings::load(*self.dimension(py), &self.word_embeddings(py), &self.elements(py));
        let index = IndexType::load(&self.index(py), &elements);

        Ok(index.count_some_neighbors(layer, start, stop))
    }

});

/// A wrapper around granne::QueryEmbeddings to allow building indexes from python
#[derive(Clone)]
pub struct MmapQueryEmbeddings {
    word_embeddings: Arc<memmap::Mmap>,
    queries: Arc<memmap::Mmap>,
    dimension: usize,
}

impl MmapQueryEmbeddings {
    pub fn new(dimension: usize, word_embeddings: memmap::Mmap, queries: memmap::Mmap) -> Self {
        Self {
            word_embeddings: Arc::new(word_embeddings),
            queries: Arc::new(queries),
            dimension: dimension,
        }
    }

    #[inline(always)]
    pub fn load<'a>(self: &'a Self) -> granne::QueryEmbeddings<'a> {
        granne::QueryEmbeddings::load(self.dimension, &self.word_embeddings[..], &self.queries[..])
    }
}

impl At for MmapQueryEmbeddings {
    type Output = granne::AngularVector<'static>;

    fn at(self: &Self, index: usize) -> Self::Output {
        self.load().at(index)
    }

    fn len(self: &Self) -> usize {
        self.load().len()
    }
}

pub fn py_compress_index(py: Python, input_path: String, output_path: String) -> PyResult<PyObject> {
    granne::compress_index(&input_path, &output_path).expect("Could not write index");

    Ok(py.None())
}

pub fn py_parse_queries_and_save_to_disk(
    py: Python,
    queries_path: String,
    words_path: String,
    output_path: String,
    show_progress: bool,
) -> PyResult<PyObject> {
    granne::query_embeddings::parsing::parse_queries_and_save_to_disk(
        &Path::new(&queries_path),
        &Path::new(&words_path),
        &Path::new(&output_path),
        show_progress,
    );

    Ok(py.None())
}

pub fn py_compute_query_vectors_and_save_to_disk(
    py: Python,
    dimension: usize,
    queries_path: String,
    word_embeddings_path: String,
    output_path: String,
    begin: usize,
    end: usize,
    dtype: DTYPE,
    show_progress: bool,
) -> PyResult<PyObject> {
    match dtype {
        DTYPE::F32 => granne::query_embeddings::parsing::compute_range_of_query_vectors_and_save_to_disk::<f32>(
            dimension,
            &Path::new(&queries_path),
            &Path::new(&word_embeddings_path),
            &Path::new(&output_path),
            begin,
            end,
            show_progress,
        ),
        DTYPE::I8 => granne::query_embeddings::parsing::compute_range_of_query_vectors_and_save_to_disk::<i8>(
            dimension,
            &Path::new(&queries_path),
            &Path::new(&word_embeddings_path),
            &Path::new(&output_path),
            begin,
            end,
            show_progress,
        ),
    };

    Ok(py.None())
}

py_class!(pub class QueryHnswBuilder |py| {
    data builder: RefCell<BuilderType>;

    @classmethod
    def with_mmapped_elements(_cls,
                              dimension: usize,
                              num_layers: usize,
                              word_embeddings_path: &str,
                              queries_path: &str,
                              index_path: Option<String> = None,
                              num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
                              max_search: usize = DEFAULT_MAX_SEARCH,
                              reinsert_elements: bool = true,
                              show_progress: bool = true) -> PyResult<QueryHnswBuilder>
    {
        let config = granne::Config {
            num_layers: num_layers,
            num_neighbors: num_neighbors,
            max_search: max_search,
            reinsert_elements: reinsert_elements,
            show_progress: show_progress
        };

        let word_embeddings = File::open(word_embeddings_path).unwrap();
        let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

        let queries = File::open(queries_path).unwrap();
        let queries = unsafe { memmap::Mmap::map(&queries).unwrap() };

        let elements = MmapQueryEmbeddings::new(dimension, word_embeddings, queries);

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

    def save_index(&self, path: &str, compress: bool = false) -> PyResult<PyObject> {
        let builder = self.builder(py).borrow();

        if compress {
            builder.save_compressed_index_to_disk(path).expect("Could not save compressed index");
        } else {
            builder.save_index_to_disk(path).expect("Could not save index to disk");
        }

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
