#[macro_use]
extern crate cpython;

extern crate memmap;
extern crate granne;

use cpython::{PyObject, PyResult};
use std::cell::RefCell;
use std::fs::File;
use memmap::Mmap;
use std::iter::FromIterator;

const DIM: usize = 100;
type Scalar = f32;
type ElementType = granne::AngularVector<[Scalar; DIM]>;

const DEFAULT_NUM_NEIGHBORS: usize = 5;
const DEFAULT_MAX_SEARCH: usize = 50;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    try!(m.add(py, "__doc__", "This module is implemented in Rust."));
    try!(m.add_class::<Hnsw>(py));
    try!(m.add_class::<HnswBuilder>(py));

    Ok(())
});

macro_rules! match_dimension_and_get_index {
    ($index:expr, $elements:expr, $dim:expr, $($dims:expr),+) => {
        {
            match $dim {
                $($dims => {
                    Box::new(PyHnsw::<granne::AngularVector<[f32; $dims]>>::new(
                        $index,
                        $elements
                    ))
                },)+
                _ => panic!("Unsupported dimension"),
            }
        }
    };
}

macro_rules! boxed_index {
    ($index:expr, $elements:expr, $dim:expr) => {
        match_dimension_and_get_index!(
            $index, $elements, $dim,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30,
            32, 50, 60, 64, 96, 100, 128, 200, 256, 300)
    };
}

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

        let _: Box<SearchIndex> = boxed_index!(
            &index, &elements, dimension);

        Hnsw::create_instance(py, index, elements, dimension)
    }

    def search(&self,
               element: Vec<f32>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let index: Box<SearchIndex> = boxed_index!(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.search(element, num_elements, max_search))
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let index: Box<SearchIndex> = boxed_index!(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.get_element(idx))
    }

    def __len__(&self) -> PyResult<usize> {
        let index: Box<SearchIndex> = boxed_index!(
            self.index(py), self.elements(py), *self.dimension(py));

        Ok(index.len())
    }

});

trait SearchIndex {
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)>;

    fn get_element(self: &Self, idx: usize) -> Vec<f32>;

    fn len(self: &Self) -> usize;
}

struct PyHnsw<'a, T: 'a + granne::ComparableTo<T>> {
    index: granne::Hnsw<'a, T, T>,
    elements: &'a [T],
}

impl<'a, T: 'a + granne::ComparableTo<T>> PyHnsw<'a, T> {
    fn new(index: &'a [u8], elements: &'a [u8]) -> Self {
        Self {
            index: granne::Hnsw::load(index, granne::file_io::load(elements)),
            elements: granne::file_io::load(elements),
        }
    }
}

impl<'a, T> SearchIndex for PyHnsw<'a, T>
    where T: 'a + granne::ComparableTo<T> + granne::Dense<f32> + FromIterator<f32> {
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)> {
        self.index.search(&element.into_iter().collect(), num_elements, max_search)
    }

    fn get_element(self: &Self, idx: usize) -> Vec<f32> {
        self.elements[idx].as_slice().to_vec()
    }

    fn len(self: &Self) -> usize {
        self.elements.len()
    }
}


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
        self.builder(py).borrow_mut().add(vec![element.into_iter().collect()]);

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
            &element.into_iter().collect(), num_neighbors, max_search))
    }

});
