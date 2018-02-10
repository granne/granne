#[macro_use]
extern crate cpython;

extern crate memmap;
extern crate granne;

use cpython::{PyObject, PyResult};
use std::cell::RefCell;
use std::fs::File;
use memmap::Mmap;
use std::iter::FromIterator;
use std::io;

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


trait IndexBuilder {
    fn add(self: &mut Self, element: Vec<f32>);
    fn build(self: &mut Self);
    fn save_to_disk(self: &Self, path: &str) -> io::Result<()>;
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)>;
    fn len(self: &Self) -> usize;
}

struct PyHnswBuilder<T: granne::ComparableTo<T> + Sync + Send> {
    builder: granne::HnswBuilder<T>,
}

impl<T> PyHnswBuilder<T>
    where T: granne::ComparableTo<T> + Sync + Send + Clone
{
    fn new(config: granne::Config) -> Self {
        Self {
            builder: granne::HnswBuilder::new(config),
        }
    }
}

impl<T> IndexBuilder for PyHnswBuilder<T>
    where T: granne::ComparableTo<T> + granne::Dense<f32> + FromIterator<f32> + Sync + Send + Clone
{
    fn add(self: &mut Self, element: Vec<f32>) {
        self.builder.add(vec![element.into_iter().collect()]);
    }

    fn build(self: &mut Self) {
        self.builder.build_index();
    }

    fn save_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.builder.save_to_disk(path)
    }

    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)>
    {
        self.builder.get_index().search(
            &element.into_iter().collect(),
            num_elements,
            max_search)
    }

    fn len(self: &Self) -> usize {
        self.builder.get_index().len()
    }
}

macro_rules! match_dimension_and_get_builder {
    ($config:expr, $dim:expr, $($dims:expr),+) => {
        {
            match $dim {
                $($dims => {
                    Box::new(PyHnswBuilder::<granne::AngularVector<[f32; $dims]>>::new(
                        $config
                    ))
                },)+
                _ => panic!("Unsupported dimension"),
            }
        }
    };
}

macro_rules! boxed_builder {
    ($config:expr, $dim:expr) => {
        match_dimension_and_get_builder!(
            $config, $dim,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30,
            32, 50, 60, 64, 96, 100, 128, 200, 256, 300)
    };
}

py_class!(class HnswBuilder |py| {
    data builder: RefCell<Box<IndexBuilder + Send>>;

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

        let builder: Box<IndexBuilder + Send> = boxed_builder!(
            config, dimension
        );

        HnswBuilder::create_instance(py, RefCell::new(builder))
    }
/*
    @classmethod def load(_cls, path: &str, element_path: &str) -> PyResult<HnswBuilder> {
        let mut file = File::open(path).unwrap();
        let mut element_file = File::open(element_path).unwrap();

        let builder = granne::HnswBuilder::read(&mut file, &mut element_file).unwrap();

        HnswBuilder::create_instance(py, RefCell::new(builder))
    }
*/

    def add(&self, element: Vec<f32>) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().add(element);

        Ok(py.None())
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().len())
    }

    def build(&self) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().build();

        Ok(py.None())
    }

    def save(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_to_disk(path).unwrap();

        Ok(py.None())
    }

    def search(&self, element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        Ok(self.builder(py).borrow().search(element, num_neighbors, max_search))
    }


});
