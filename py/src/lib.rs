#[macro_use]
extern crate cpython;

extern crate memmap;
extern crate granne;

use cpython::{PyObject, PyResult};
use std::cell::RefCell;
use std::fs::File;
use memmap::Mmap;

const DEFAULT_NUM_NEIGHBORS: usize = 5;
const DEFAULT_MAX_SEARCH: usize = 50;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    try!(m.add(py, "__doc__", "granne - Graph-based Retrieval of Approximate Nearest Neighbors"));
    try!(m.add_class::<Hnsw>(py));
    try!(m.add_class::<HnswBuilder>(py));

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


py_class!(class HnswBuilder |py| {
    data builder: RefCell<Box<granne::IndexBuilder + Send>>;

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

        let builder: Box<granne::IndexBuilder + Send> = granne::boxed_index_builder(
            "f32", dimension, config, None
        );

        HnswBuilder::create_instance(py, RefCell::new(builder))
    }
/*
    @classmethod
    def with_elements(_cls,
                      elements_path: &str,
                      dimension: usize,
                      num_layers: usize,
                      max_search: usize = DEFAULT_MAX_SEARCH,
                      show_progress: bool = true) -> PyResult<HnswBuilder> {

        let elements = File::open(elements_path).unwrap();
        let elements = unsafe { Mmap::map(&elements).unwrap() };

        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let builder: Box<granne::IndexBuilder + Send> = granne::boxed_index_builder(
            "f32", dimension, config, Some(&elements[..])
        );

        HnswBuilder::create_instance(py, RefCell::new(builder), Some(elements))
    }
*/

    def add(&self, element: Vec<f32>) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().add(element);

        Ok(py.None())
    }

    def __len__(&self) -> PyResult<usize> {
        let builder = self.builder(py).borrow();
        let index = builder.get_index();
        Ok(index.len())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let builder = self.builder(py).borrow();
        let index = builder.get_index();

        Ok(index.get_element(idx))
    }

    def build(&self) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().build();

        Ok(py.None())
    }

    def save_index(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_index_to_disk(path).unwrap();

        Ok(py.None())
    }

    def save_elements(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_elements_to_disk(path).unwrap();

        Ok(py.None())
    }

    def search(&self, element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let builder = self.builder(py).borrow();
        let index = builder.get_index();
        Ok(index.search(element, num_neighbors, max_search))
    }


});
