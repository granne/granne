#[macro_use]
extern crate cpython;

use cpython::{PyObject, PyResult, Python};
use granne;
use madvise::{AccessPattern, AdviseMemory};
use memmap::Mmap;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    m.add(
        py,
        "__doc__",
        "granne - Graph-based Retrieval of Approximate Nearest Neighbors",
    )?;
    m.add_class::<Granne>(py)?;
    m.add_class::<GranneBuilder>(py)?;

    Ok(())
});

const DEFAULT_NUM_LAYERS: usize = 7;
const DEFAULT_NUM_NEIGHBORS: usize = 20;
const DEFAULT_MAX_SEARCH: usize = 200;

#[derive(Clone, Copy)]
pub enum ElementType {
    Angular,
    AngularInt,
    Embeddings,
}

pub enum MyElements {
    Angular((memmap::Mmap, usize)),
    AngularInt((memmap::Mmap, usize)),
    Embeddings((memmap::Mmap, memmap::Mmap, usize)),
}

impl Default for ElementType {
    fn default() -> Self {
        Self::Angular
    }
}

impl ElementType {
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "angular" => ElementType::Angular,
            "angular_int" => ElementType::AngularInt,
            "embeddings" | "sum_embeddings" | "sumembeddings" => ElementType::Embeddings,
            _ => panic!("Invalid element_type"),
        }
    }
}

impl<'source> cpython::FromPyObject<'source> for ElementType {
    fn extract(py: Python, obj: &'source PyObject) -> PyResult<Self> {
        let vector_type_string: String = String::extract(py, obj)?;

        Ok(ElementType::from_str(&vector_type_string))
    }
}

py_class!(class Granne |py| {
    data index: memmap::Mmap;
    data elements: MyElements;

    def __new__(_cls,
                index_path: String,
                element_type: ElementType,
                elements_path: String,
                dimension: usize,
                embeddings_path: Option<String>
    ) -> PyResult<Granne> {

        let index = File::open(index_path).unwrap();
        let index = unsafe { Mmap::map(&index).unwrap() };
        index.advise_memory_access(AccessPattern::Random).expect("Error with madvise");


        let elements = File::open(elements_path).unwrap();
        let elements = unsafe { Mmap::map(&elements).unwrap() };
        elements.advise_memory_access(AccessPattern::Random).expect("Error with madvise");

        let elements = match element_type {
            ElementType::Angular => MyElements::Angular((elements, dimension)),
            ElementType::AngularInt => MyElements::AngularInt((elements, dimension)),
            ElementType::Embeddings => {
                let embeddings = File::open(embeddings_path.expect("Embeddings are required for this element type")).unwrap();
                let embeddings = unsafe { Mmap::map(&embeddings).unwrap() };
                embeddings.advise_memory_access(AccessPattern::Random).expect("Error with madvise");

                MyElements::Embeddings((elements, embeddings, dimension))
            },
        };

        let _ = boxed_index(&index[..], &elements);

        Granne::create_instance(py, index, elements)
    }

    def search(&self,
               element: Vec<f32>,
               max_search: usize = DEFAULT_MAX_SEARCH,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS) -> PyResult<Vec<(usize, f32)>>
    {
        let index = boxed_index(self.index(py), self.elements(py));

        Ok(index.search(element, num_elements, max_search))
    }

    def search_batch(&self,
                     elements: Vec<Vec<f32>>,
                     num_elements: usize = DEFAULT_NUM_NEIGHBORS,
                     max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<Vec<(usize, f32)>>>
    {
        let index = boxed_index(self.index(py), self.elements(py));

        Ok(elements
           .into_par_iter()
           .map(|element| index.search(element, num_elements, max_search))
           .collect()
        )
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let index = boxed_index(self.index(py), self.elements(py));

        Ok(index.get_element(idx).into())
    }

    def __len__(&self) -> PyResult<usize> {
        let index = boxed_index(self.index(py), self.elements(py));

        Ok(index.len())
    }

    def get_neighbors(&self, idx: usize, layer: usize) -> PyResult<Vec<usize>> {
        let index = boxed_index(self.index(py), self.elements(py));

        Ok(index.get_neighbors(idx, layer))
    }
});

py_class!(class GranneBuilder |py| {
    data builder: RefCell<Box<dyn Builder + Send + Sync>>;

    def __new__(_cls,
                element_type: ElementType,
                elements_path: Option<String> = None,
                dimension: usize = 0,
                embeddings_path: Option<String> = None,
                num_layers: usize = DEFAULT_NUM_LAYERS, // replace with layer multiplier
                num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
                max_search: usize = DEFAULT_MAX_SEARCH,
                reinsert_elements: bool = true,
                expected_size: usize = 0,
                index_path: Option<String> = None,
                show_progress: bool = true) -> PyResult<GranneBuilder> {

        let config = granne::Config {
            num_layers: num_layers,
            num_neighbors: num_neighbors,
            max_search: max_search,
            reinsert_elements: reinsert_elements,
            show_progress: show_progress
        };

        let builder: Box<dyn Builder + Send + Sync> =
            if let Some(path) = elements_path {
                assert!(dimension > 0);
                match element_type {
                    ElementType::Angular => {
                        Box::new(granne::GranneBuilder::new(
                            config,
                            granne::angular::mmap::MmapVectors::new(&path, dimension)
                        ))
                    },
                    ElementType::AngularInt => {
                        Box::new(granne::GranneBuilder::new(
                            config,
                            granne::angular_int::mmap::MmapVectors::new(&path, dimension)
                        ))
                    },
                    ElementType::Embeddings => {
                        let elements = granne::sum_embeddings::mmap::MmapSumEmbeddings::new(
                            &path,
                            &embeddings_path.expect("Embeddings are required for this element type"),
                            dimension);

                        Box::new(granne::GranneBuilder::new(
                            config,
                            elements,
                        ))
                    }
                    
                }
            } else {
                match element_type {
                    ElementType::Angular => {
                        Box::new(granne::GranneBuilder::new(
                            config,
                            granne::angular::Vectors::new(),
                        ))
                    },
                    ElementType::AngularInt => {
                        Box::new(granne::GranneBuilder::new(
                            config,
                            granne::angular_int::Vectors::new(),
                        ))
                    },
                    _ => { panic!("An empty GranneBuilder is not supported for this element type!") },
                }
            };

        GranneBuilder::create_instance(py, RefCell::new(builder))
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().len())
    }

    def indexed_elements(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().indexed_elements())
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        Ok(self.builder(py).borrow().get_element(idx))
    }

    def add(&self, element: Vec<f32>) -> PyResult<PyObject> {
        Ok(py.None())
    }

    def build_index(&self, num_elements: usize = <usize>::max_value()) -> PyResult<PyObject> {
        let num_elements = std::cmp::min(num_elements, self.builder(py).borrow().len());

        self.builder(py).borrow_mut().build_index(num_elements);

        Ok(py.None())
    }

    def save_index(&self, path: &str, compress: bool = false) -> PyResult<PyObject> {
        self.builder(py).borrow().save_index(path).expect(&format!("Could not save index to {}", path));

        Ok(py.None())
    }

    def save_elements(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_elements(path).expect(&format!("Could not save elements to {}", path));

        Ok(py.None())
    }

    def search(&self,
               element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        Ok(self.builder(py).borrow().search(element, num_neighbors, max_search))
    }
});

/*
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
        let index_elements = self.shards(py).iter().map(|&(ref a, ref b)| (a, granne::AngularVectors::load(100, &b[..]))).collect::<Vec<_>>();
        let index: granne::ShardedHnsw<granne::AngularVectors, granne::AngularVector> = granne::ShardedHnsw::new(
            &index_elements.iter().map(|&(ref a, ref b)| (&a[..], b)).collect::<Vec<_>>()[..]
        );

        Ok(index.search(
            &element.into(), num_neighbors, max_search))
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {

        let index_elements = self.shards(py).iter().map(|&(ref a, ref b)| (a, granne::AngularVectors::load(100, &b[..]))).collect::<Vec<_>>();
        let index: granne::ShardedHnsw<granne::AngularVectors, granne::AngularVector> = granne::ShardedHnsw::new(
            &index_elements.iter().map(|&(ref a, ref b)| (&a[..], b)).collect::<Vec<_>>()[..]
        );

        Ok(index.get_element(idx).into())
    }

    def __len__(&self) -> PyResult<usize> {
        let index_elements = self.shards(py).iter().map(|&(ref a, ref b)| (a, granne::AngularVectors::load(100, &b[..]))).collect::<Vec<_>>();
        let index: granne::ShardedHnsw<granne::AngularVectors, granne::AngularVector> = granne::ShardedHnsw::new(
            &index_elements.iter().map(|&(ref a, ref b)| (&a[..], b)).collect::<Vec<_>>()[..]
        );

        Ok(index.len())
    }

});
*/

trait Index {
    fn search(
        self: &Self,
        element: Vec<f32>,
        max_search: usize,
        num_elements: usize,
    ) -> Vec<(usize, f32)>;
    fn get_element(self: &Self, idx: usize) -> Vec<f32>;
    fn len(self: &Self) -> usize;
    fn get_neighbors(self: &Self, idx: usize, layer: usize) -> Vec<usize>;
}

impl<Elements> Index for granne::Granne<'_, Elements>
where
    Elements: granne::ElementContainer,
    Elements::Element: From<Vec<f32>>,
{
    fn search(
        self: &Self,
        element: Vec<f32>,
        max_search: usize,
        num_elements: usize,
    ) -> Vec<(usize, f32)> {
        self.search(&element.into(), max_search, num_elements)
    }

    fn get_element(self: &Self, idx: usize) -> Vec<f32> {
        panic!("get_element not yet implemented!")
    }

    fn len(self: &Self) -> usize {
        Self::len(self)
    }

    fn get_neighbors(self: &Self, idx: usize, layer: usize) -> Vec<usize> {
        Self::get_neighbors(self, idx, layer)
    }
}

fn boxed_index<'a>(index: &'a [u8], elements: &'a MyElements) -> Box<dyn Index + Sync + 'a> {
    match elements {
        MyElements::Angular((elements, dim)) => {
            let elements = granne::angular::Vectors::load(elements, *dim);
            Box::new(granne::Granne::load(index, elements))
        }
        MyElements::AngularInt((elements, dim)) => {
            let elements = granne::angular_int::Vectors::load(elements, *dim);
            Box::new(granne::Granne::load(index, elements))
        }
        MyElements::Embeddings((elements, embeddings, dim)) => {
            let embeddings = granne::angular::Vectors::load(embeddings, *dim);
            let elements = granne::sum_embeddings::SumEmbeddings::load(embeddings, elements);

            Box::new(granne::Granne::load(index, elements))
        }
    }
}

trait Builder: Index {
    fn indexed_elements(self: &Self) -> usize;
    fn build_index(self: &mut Self, num_elements: usize);
    fn save_index(self: &Self, path: &str) -> std::io::Result<()>;
    fn save_elements(self: &Self, path: &str) -> std::io::Result<()>;
}

impl<Elements> Builder for granne::GranneBuilder<Elements>
where
    Elements: granne::ElementContainer + granne::Writeable + Sync,
    Elements::Element: From<Vec<f32>>,
{
    fn indexed_elements(self: &Self) -> usize {
        Self::indexed_elements(self)
    }

    fn build_index(self: &mut Self, num_elements: usize) {
        Self::build_index_part(self, num_elements)
    }

    fn save_index(self: &Self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        self.write_index(&mut file)
    }

    fn save_elements(self: &Self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        self.write_elements(&mut file).map(|_| {})
    }
}

impl<Elements: granne::ElementContainer + Sync> Index for granne::GranneBuilder<Elements>
where
    Elements::Element: From<Vec<f32>>,
{
    fn search(
        self: &Self,
        element: Vec<f32>,
        max_search: usize,
        num_elements: usize,
    ) -> Vec<(usize, f32)> {
        Index::search(&self.get_index(), element, max_search, num_elements)
    }

    fn get_element(self: &Self, idx: usize) -> Vec<f32> {
        Index::get_element(&self.get_index(), idx)
    }

    fn len(self: &Self) -> usize {
        Index::len(&self.get_index())
    }

    fn get_neighbors(self: &Self, idx: usize, layer: usize) -> Vec<usize> {
        Index::get_neighbors(&self.get_index(), idx, layer)
    }
}
