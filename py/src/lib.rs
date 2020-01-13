#[macro_use]
extern crate cpython;

use cpython::{PyObject, PyResult, Python};
use granne::{self, Builder};
use std::cell::RefCell;

mod variants;

const DEFAULT_MAX_SEARCH: usize = 200;
const DEFAULT_NUM_NEIGHBORS: usize = 30;

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    m.add(
        py,
        "__doc__",
        "Graph-based Retrieval of Approximate Nearest Neighbors",
    )?;
    m.add_class::<Granne>(py)?;
    m.add_class::<GranneBuilder>(py)?;
    Ok(())
});

py_class!(class Granne |py| {
    data index: RefCell<Box<dyn PyGranne + Send + Sync>>;

    def __new__(_cls,
                index_path: String,
                element_type: String,
                elements_path: String,
                dimension: Option<usize> = None,
                embeddings_path: Option<String> = None,
                words_path: Option<String> = None
    ) -> PyResult<Granne> {

        let index: Box<dyn PyGranne + Send + Sync> = match element_type.to_ascii_lowercase().as_str() {
            "angular" => Box::new(variants::index::AngularGranne::new(
                &index_path,
                &elements_path,
                dimension.expect("dimension missing!"),
            )),
            "angular_int" => Box::new(variants::index::AngularIntGranne::new(
                &index_path,
                &elements_path,
                dimension.expect("dimension missing!"),
            )),
            "embeddings" => Box::new(variants::index::WordEmbeddingsGranne::new(
                &index_path,
                &elements_path,
                &embeddings_path.expect("embeddings_path required for this element type!"),
                &words_path.expect("words_path required for this element type!"),
            )),
            _ => panic!("Invalid element type"),
        };

        Granne::create_instance(py, RefCell::new(index))
    }

    /// Searches for nearest neighbors to an element. The type of element depends on the element type of this index.
    def search(&self,
               element: &PyObject,
               max_search: usize = DEFAULT_MAX_SEARCH,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS) -> PyResult<Vec<(usize, f32)>>
    {
        self.index(py).borrow().search(py, element, max_search, num_elements)
    }

    /// Returns the element at index idx.
    def get_element(&self, idx: usize) -> PyResult<PyObject> {
        Ok(self.index(py).borrow().get_element(py, idx))
    }

    /// Returns the internal element at index idx (may be the same as the element at idx).
    def get_internal_element(&self, idx: usize) -> PyResult<PyObject> {
        Ok(self.index(py).borrow().get_internal_element(py, idx))
    }

    /// Returns the neighbors of the element at idx in layer (default: last layer) in the HNSW graph.
    def get_neighbors(&self, idx: usize, layer: Option<usize> = None) -> PyResult<Vec<usize>> {
        if let Some(layer) = layer {
            Ok(self.index(py).borrow().get_neighbors(idx, layer))
        } else {
            let index = self.index(py).borrow();
            Ok(index.get_neighbors(idx, index.num_layers() - 1))
        }
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.index(py).borrow().len())
    }

    /// Returns the number of layers in this index..
    def num_layers(&self) -> PyResult<usize> {
        Ok(self.index(py).borrow().num_layers())
    }

    /// Returns the number of elements in layer.
    def layer_len(&self, layer: usize) -> PyResult<usize> {
        Ok(self.index(py).borrow().layer_len(layer))
    }
});

py_class!(class GranneBuilder |py| {
    data builder: RefCell<Box<dyn PyGranneBuilder + Send + Sync>>;

    def __new__(_cls,
                element_type: String,
                elements_path: Option<String> = None,
                dimension: Option<usize> = None,
                embeddings_path: Option<String> = None,
                words_path: Option<String> = None,
                index_path: Option<String> = None,
                layer_multiplier: Option<f32> = None,
                expected_num_elements: Option<usize> = None,
                num_neighbors: Option<usize> = None,
                max_search: Option<usize> = None,
                reinsert_elements: bool = true,
                show_progress: bool = true) -> PyResult<GranneBuilder> {

        let mut config = granne::BuildConfig::default()
            .show_progress(show_progress)
            .reinsert_elements(reinsert_elements);

        // some unnecessary clones
        config = layer_multiplier.map_or(config.clone(), |x| config.layer_multiplier(x));
        config = expected_num_elements.map_or(config.clone(), |x| config.expected_num_elements(x));
        config = num_neighbors.map_or(config.clone(), |x| config.num_neighbors(x));
        config = max_search.map_or(config.clone(), |x| config.max_search(x));

        let builder: Box<dyn PyGranneBuilder + Send + Sync> = match (
            elements_path.as_ref(),
            element_type.to_ascii_lowercase().as_str(),
        ) {
            (None, "angular") => Box::new(granne::GranneBuilder::new(
                config,
                granne::angular::Vectors::new(),
            )),
            (Some(path), "angular") => Box::new(granne::GranneBuilder::new(
                config,
                granne::angular::mmap::MmapVectors::new(path, dimension.expect("dimension missing")),
            )),
            (None, "angular_int") => Box::new(granne::GranneBuilder::new(
                config,
                granne::angular_int::Vectors::new(),
            )),
            (Some(path), "angular_int") => Box::new(granne::GranneBuilder::new(
                config,
                granne::angular_int::mmap::MmapVectors::new(
                    path,
                    dimension.expect("dimension missing"),
                ),
            )),
            (Some(path), "embeddings") => {
                Box::new(variants::builder::WordEmbeddingsBuilder::new(
                    config,
                    path,
                    &embeddings_path.expect("embeddings_path required for this element type!"),
                    &words_path.expect("words_path required for this element type!"),
                ))
            }
            _ => panic!(),
        };

        GranneBuilder::create_instance(py, RefCell::new(builder))
    }

/*
    def get_element(&self, idx: usize) -> PyResult<PyObject> {
        Ok(self.builder(py).borrow().get_index().get_element(py, idx))
    }
     */

    /// Append one element to this builder. Note: the element will not be indexed until GranneBuilder.build() is called.
    def append(&self, element: &PyObject) -> PyResult<PyObject> {
        self.builder(py).borrow_mut().push(py, element)
    }

    /// Builds an index with the first num_elements elements (or all if not specified).
    def build(&self, num_elements: usize = <usize>::max_value()) -> PyResult<PyObject> {
        if num_elements == usize::max_value() {
            self.builder(py).borrow_mut().build();
        } else {
            self.builder(py).borrow_mut().build_partial(num_elements);
        }

        Ok(py.None())
    }

    /// Saves the index to a file.
    def save_index(&self, path: &str/*, compress: bool = false*/) -> PyResult<PyObject> {
        self.builder(py).borrow().save_index(path).expect(&format!("Could not save index to {}", path));

        Ok(py.None())
    }

    /// Saves the elements to a file.
    def save_elements(&self, path: &str) -> PyResult<PyObject> {
        self.builder(py).borrow().save_elements(path).expect(&format!("Could not save elements to {}", path));

        Ok(py.None())
    }
/*
    def search(&self,
               element: &PyObject,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        self.builder(py).borrow().get_index().search(py, element, num_neighbors, max_search)
    }
     */

    /// Returns the neighbors of the element at idx in layer in the HNSW graph.
    def get_neighbors(&self, idx: usize, layer: Option<usize> = None) -> PyResult<Vec<usize>> {
        if let Some(layer) = layer {
            Ok(self.builder(py).borrow().get_neighbors(idx, layer))
        } else {
            let index = self.builder(py).borrow();
            Ok(index.get_neighbors(idx, index.num_layers() - 1))
        }
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().len())
    }

    /// Returns the number of elements in this builder.
    def num_elements(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().num_elements())
    }

    /// Returns the number of layers in this index..
    def num_layers(&self) -> PyResult<usize> {
        Ok(self.builder(py).borrow().num_layers())
    }

    /// Returns the number of elements in layer.
    def layer_len(&self, layer: usize) -> PyResult<usize> {
        Ok(self.builder(py).borrow().layer_len(layer))
    }
});

trait PyGranne: granne::Index {
    fn search(
        self: &Self,
        py: Python,
        element: &PyObject,
        max_search: usize,
        num_elements: usize,
    ) -> PyResult<Vec<(usize, f32)>>;
    fn get_element(self: &Self, py: Python, idx: usize) -> PyObject;
    fn get_internal_element(self: &Self, py: Python, idx: usize) -> PyObject {
        self.get_element(py, idx)
    }
}

trait SaveIndex: granne::Builder {
    fn save_index(self: &Self, path: &str) -> std::io::Result<()>;

    fn save_elements(self: &Self, path: &str) -> std::io::Result<()>;
}

impl<Elements: granne::ElementContainer + granne::Writeable + Sync> SaveIndex
    for granne::GranneBuilder<Elements>
{
    fn save_index(self: &Self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_index(&mut file)
    }

    fn save_elements(self: &Self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_elements(&mut file).map(|_| {})
    }
}

trait PyGranneBuilder: granne::Builder + granne::Index + SaveIndex {
    fn push(self: &mut Self, _py: Python, _element: &PyObject) -> PyResult<PyObject> {
        panic!("Not possible with this builder type!");
    }
}
