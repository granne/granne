#[macro_use]
extern crate cpython;

extern crate memmap;
extern crate granne;

use cpython::{PyObject, PyResult};
use std::cell::RefCell;
use std::fs::File;
use memmap::Mmap;

type Scalar = i8;
type ElementType = granne::Int8Element;

const DEFAULT_NUM_NEIGHBORS: usize = 5;
const DEFAULT_MAX_SEARCH: usize = 50;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    try!(m.add(py, "__doc__", "This module is implemented in Rust."));
    try!(m.add_class::<Hnsw>(py));
    try!(m.add_class::<HnswBuilder>(py));

    Ok(())
});

py_class!(class Hnsw |py| {
    data index: memmap::Mmap;
    data elements: memmap::Mmap;

    def __new__(_cls,
                index_path: &str,
                elements_path: &str) -> PyResult<Hnsw> {

        let index = File::open(index_path).unwrap();
        let index = unsafe { Mmap::map(&index).unwrap() };

        let elements = File::open(elements_path).unwrap();
        let elements = unsafe { Mmap::map(&elements).unwrap() };

        Hnsw::create_instance(py, index, elements)
    }

    def search(&self,
               element: Vec<Scalar>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let index = granne::Hnsw::<ElementType, ElementType>::load(&self.index(py), granne::file_io::load(&self.elements(py)));

        Ok(index.search(
            &element.into(), num_elements, max_search))
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<Scalar>> {
        let elements = granne::file_io::load::<ElementType>(&self.elements(py));
        Ok(elements[idx].0.to_vec())
    }

    def __len__(&self) -> PyResult<usize> {
        let elements = granne::file_io::load::<ElementType>(&self.elements(py));
        Ok(elements.len())
    }

});


py_class!(class HnswBuilder |py| {
    data builder: RefCell<granne::HnswBuilder<ElementType>>;

    def __new__(_cls,
                num_layers: usize,
                max_search: usize = DEFAULT_MAX_SEARCH,
                show_progress: bool = true) -> PyResult<HnswBuilder> {

        let config = granne::Config {
            num_layers: num_layers,
            max_search: max_search,
            show_progress: show_progress
        };

        let builder = granne::HnswBuilder::new(config);

        HnswBuilder::create_instance(py, RefCell::new(builder))
    }

    @classmethod def load(_cls, path: &str, element_path: &str) -> PyResult<HnswBuilder> {
        let mut file = File::open(path).unwrap();
        let mut element_file = File::open(element_path).unwrap();

        let builder = granne::HnswBuilder::read(&mut file, &mut element_file).unwrap();

        HnswBuilder::create_instance(py, RefCell::new(builder))
    }

    def add(&self, element: Vec<Scalar>) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().add(vec![element.into()]);

        Ok(py.None())
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().get_index().len())
    }

    def build(&self) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().build_index();

        Ok(py.None())
    }

    def save(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_to_disk(path).unwrap();

        Ok(py.None())
    }

    def search(&self, element: Vec<Scalar>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let builder = self.builder(py).borrow();
        let search_index = builder.get_index();

        Ok(search_index.search(
            &element.into(), num_neighbors, max_search))
    }

});
