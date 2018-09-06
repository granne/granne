#[macro_use]
extern crate cpython;

extern crate memmap;
extern crate granne;
extern crate madvise;
extern crate serde_json;

use cpython::{PyObject, PyResult};
use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;
use memmap::Mmap;
use madvise::{AccessPattern, AdviseMemory};

mod query_embeddings;
use query_embeddings::{py_parse_queries_and_save_to_disk, py_compute_query_vectors_and_save_to_disk};

const DEFAULT_NUM_NEIGHBORS: usize = 5;
const DEFAULT_MAX_SEARCH: usize = 50;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    try!(m.add(py, "__doc__", "granne - Graph-based Retrieval of Approximate Nearest Neighbors"));
    try!(m.add_class::<Hnsw>(py));
    try!(m.add_class::<ShardedHnsw>(py));
    try!(m.add_class::<HnswBuilder>(py));
    try!(m.add_class::<query_embeddings::QueryHnsw>(py));
    try!(m.add_class::<query_embeddings::QueryHnswBuilder>(py));
    try!(m.add(py, "parse_queries_and_save_to_disk",
               py_fn!(py, py_parse_queries_and_save_to_disk(
                   queries_path: String,
                   words_path: String,
                   output_path: String,
                   show_progress: bool = true)
               )
    ));
    try!(m.add(py, "compute_query_vectors_and_save_to_disk",
               py_fn!(py, py_compute_query_vectors_and_save_to_disk(
                   queries_path: String,
                   word_embeddings_path: String,
                   output_path: String,
                   show_progress: bool = true)
               )
    ));

    Ok(())
});


py_class!(class Hnsw |py| {
    data index: memmap::Mmap;
    data elements: memmap::Mmap;
    data dimension: usize;

    def __new__(_cls,
                index_path: &str,
                elements_path: &str,
                dimension: usize) -> PyResult<Hnsw> {

        let index = File::open(index_path).unwrap();
        let index = unsafe { Mmap::map(&index).unwrap() };
        index.advise_memory_access(AccessPattern::Random).expect("Error with madvise");

        let elements = File::open(elements_path).unwrap();
        let elements = unsafe { Mmap::map(&elements).unwrap() };

        let _: Box<granne::SearchIndex> = granne::boxed_index(
            &index, &elements, dimension);

        Hnsw::create_instance(py, index, elements, dimension)
    }

    def search(&self,
               element: Vec<f32>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let index: Box<granne::SearchIndex> = granne::boxed_index(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.search(element, num_elements, max_search))
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let index: Box<granne::SearchIndex> = granne::boxed_index(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.get_element(idx))
    }

    def __len__(&self) -> PyResult<usize> {
        let index: Box<granne::SearchIndex> = granne::boxed_index(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.len())
    }
});


py_class!(class ShardedHnsw |py| {
    data shards: Vec<(memmap::Mmap, memmap::Mmap)>;

    def __new__(_cls,
                index_paths: Vec<String>,
                elements_paths: Vec<String>) -> PyResult<ShardedHnsw> {

        let mmaps = index_paths.iter()
            .zip(elements_paths.iter())
            .map(&|(index_path, elements_path)| {
                let index = File::open(index_path).unwrap();
                let index = unsafe { Mmap::map(&index).unwrap() };

                let elements = File::open(elements_path).unwrap();
                let elements = unsafe { Mmap::map(&elements).unwrap() };

                (index, elements)
            }).collect();

        ShardedHnsw::create_instance(py, mmaps)
    }

    def search(&self,
               element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {

        let index = granne::boxed_sharded_index(&self.shards(py).iter().map(|&(ref a, ref b)| (&a[..], &b[..])).collect::<Vec<_>>(), 100);

        Ok(index.search(
            element, num_neighbors, max_search))
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {

        let index = granne::boxed_sharded_index(&self.shards(py).iter().map(|&(ref a, ref b)| (&a[..], &b[..])).collect::<Vec<_>>(), 100);

        Ok(index.get_element(idx))
    }

    def __len__(&self) -> PyResult<usize> {
        let index = granne::boxed_sharded_index(&self.shards(py).iter().map(|&(ref a, ref b)| (&a[..], &b[..])).collect::<Vec<_>>(), 100);

        Ok(index.len())
    }

});


py_class!(class HnswBuilder |py| {
    data builder: RefCell<Box<granne::IndexBuilder + Send>>;
    data dimension: usize;
    data elements: RefCell<Option<Vec<f32>>>;
    data config: granne::Config;

    def __new__(_cls,
                dimension: usize,
                num_layers: usize,
                max_search: usize = DEFAULT_MAX_SEARCH,
                show_progress: bool = true) -> PyResult<HnswBuilder> {

        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let builder = granne::boxed_builder(
            dimension, config.clone()
        );

        HnswBuilder::create_instance(py, RefCell::new(builder), dimension, RefCell::new(Some(Vec::new())), config)
    }

    @classmethod
    def with_owned_elements(
        _cls,
        dimension: usize,
        num_layers: usize,
        elements_path: &str,
        index_path: Option<String> = None,
        max_search: usize = DEFAULT_MAX_SEARCH,
        show_progress: bool = true) -> PyResult<HnswBuilder>
    {
        let elements = File::open(elements_path).expect("Could not open elements file");
        let elements = unsafe { Mmap::map(&elements).unwrap() };

        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let index = if let Some(index_path) = index_path {
            let index_file = File::open(index_path).expect("Could not open index file");
            let index_file = BufReader::new(index_file);

            Some(index_file)

        } else {
            None
        };

        let builder = granne::boxed_owning_builder(
            dimension, config.clone(), &elements[..], index);

        HnswBuilder::create_instance(py, RefCell::new(builder), dimension, RefCell::new(None), config)
    }

    @classmethod
    def with_mmapped_elements(
        _cls,
        dimension: usize,
        num_layers: usize,
        elements_path: &str,
        index_path: Option<String> = None,
        max_search: usize = DEFAULT_MAX_SEARCH,
        show_progress: bool = true) -> PyResult<HnswBuilder>
    {
        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let index = if let Some(index_path) = index_path {
            let index_file = File::open(index_path).expect("Could not open index file");
            let index_file = BufReader::new(index_file);

            Some(index_file)

        } else {
            None
        };


        let builder = granne::boxed_mmap_builder(
            dimension, config.clone(), elements_path, index);

        HnswBuilder::create_instance(py, RefCell::new(builder), dimension, RefCell::new(None), config)
    }


    def add(&self, element: Vec<f32>) -> PyResult<PyObject> {
        assert_eq!(*self.dimension(py), element.len());

        if let Some(ref mut elements) = *self.elements(py).borrow_mut() {
            elements.extend(element);

            return Ok(py.None())

        } else {
            panic!("Cannot add (more) elements to this builder!");
        }
    }

    def __len__(&self) -> PyResult<usize> {
        if let Some(ref elements) = *self.elements(py).borrow() {
            Ok(elements.len() / self.dimension(py))

        } else {
            let builder = self.builder(py).borrow();

            Ok(builder.len())
        }
    }

    def indexed_elements(&self) -> PyResult<usize> {
        let builder = self.builder(py).borrow();

        Ok(builder.indexed_elements())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        if let Some(elements) = self.elements(py).replace(None) {
            let bytes: &[u8] = unsafe {
                ::std::slice::from_raw_parts(
                    elements.as_ptr() as *const u8,
                    elements.len() * ::std::mem::size_of::<f32>(),
                )
            };

            let mut builder = granne::boxed_owning_builder(*self.dimension(py), self.config(py).clone(), bytes, None);

            self.builder(py).replace(builder);
        }

        let builder = self.builder(py).borrow();
        let index = builder.get_index();

        return Ok(index.get_element(idx))
    }


    def build_index(&self, num_elements: usize = <usize>::max_value()) -> PyResult<PyObject> {
        if let Some(elements) = self.elements(py).replace(None) {
            let bytes: &[u8] = unsafe {
                ::std::slice::from_raw_parts(
                    elements.as_ptr() as *const u8,
                    elements.len() * ::std::mem::size_of::<f32>(),
                )
            };

            let mut builder = granne::boxed_owning_builder(*self.dimension(py), self.config(py).clone(), bytes, None);

            self.builder(py).replace(builder);
        }

        let mut builder = self.builder(py).borrow_mut();
        if num_elements == <usize>::max_value() {
            builder.build_index();
        } else {
            builder.build_index_part(num_elements);
        }

        Ok(py.None())
    }


    def save_index(&self, path: &str) -> PyResult<PyObject> {
        if self.elements(py).borrow().is_none() {
            let builder = self.builder(py).borrow();
            builder.save_index_to_disk(path).expect("Could not save index to disk");

            return Ok(py.None())
        } else {
            panic!("Cannot save an unbuilt index!");
        }
    }

    def save_elements(&self, path: &str) -> PyResult<PyObject> {
        if let Some(elements) = self.elements(py).replace(None) {
            let bytes: &[u8] = unsafe {
                ::std::slice::from_raw_parts(
                    elements.as_ptr() as *const u8,
                    elements.len() * ::std::mem::size_of::<f32>(),
                )
            };

            let mut builder = granne::boxed_owning_builder(*self.dimension(py), self.config(py).clone(), bytes, None);

            self.builder(py).replace(builder);
        }

        let builder = self.builder(py).borrow();

        builder.save_elements_to_disk(path).expect("Could not save elements to disk");

        Ok(py.None())
    }

    def search(&self, element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        if self.elements(py).borrow().is_none() {
            let builder = self.builder(py).borrow();
            let index = builder.get_index();

            return Ok(index.search(element, num_neighbors, max_search))
        } else {
            panic!("Cannot search an unbuilt index!");
        }
    }

});
