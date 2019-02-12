#[macro_use]
extern crate cpython;

extern crate granne;
extern crate madvise;
extern crate memmap;
extern crate rayon;
extern crate serde_json;

use cpython::{PyObject, PyResult, Python};
use madvise::{AccessPattern, AdviseMemory};
use memmap::Mmap;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;

mod query_embeddings;
use query_embeddings::{
    py_compress_index, py_compute_query_vectors_and_save_to_disk, py_parse_queries_and_save_to_disk,
};

const DEFAULT_NUM_NEIGHBORS: usize = 20;
const DEFAULT_MAX_SEARCH: usize = 200;

pub enum DTYPE {
    F32,
    I8,
}

impl DTYPE {
    fn default() -> DTYPE {
        DTYPE::F32
    }

    fn from_str(s: &str) -> DTYPE {
        match s {
            "f32" | "F32" => DTYPE::F32,
            "i8" | "I8" => DTYPE::I8,
            _ => panic!("Invalid dtype"),
        }
    }
}

impl<'source> cpython::FromPyObject<'source> for DTYPE {
    fn extract(py: Python, obj: &'source PyObject) -> PyResult<Self> {
        let dtype_string: String = String::extract(py, obj)?;

        Ok(DTYPE::from_str(&dtype_string))
    }
}

py_module_initializer!(granne, initgranne, PyInit_granne, |py, m| {
    try!(m.add(
        py,
        "__doc__",
        "granne - Graph-based Retrieval of Approximate Nearest Neighbors"
    ));
    try!(m.add_class::<Hnsw>(py));
    try!(m.add_class::<ShardedHnsw>(py));
    try!(m.add_class::<HnswBuilder>(py));
    try!(m.add_class::<query_embeddings::Embeddings>(py));
    try!(m.add_class::<query_embeddings::QueryHnsw>(py));
    try!(m.add_class::<query_embeddings::QueryHnswBuilder>(py));
    try!(m.add(
        py,
        "compress_index",
        py_fn!(py, py_compress_index(input_path: String, output_path: String))
    ));
    try!(m.add(
        py,
        "parse_queries_and_save_to_disk",
        py_fn!(
            py,
            py_parse_queries_and_save_to_disk(
                queries_path: String,
                words_path: String,
                output_path: String,
                num_shards: usize = 1,
                show_progress: bool = true
            )
        )
    ));
    try!(m.add(
        py,
        "compute_query_vectors_and_save_to_disk",
        py_fn!(
            py,
            py_compute_query_vectors_and_save_to_disk(
                dimension: usize,
                queries_path: String,
                word_embeddings_path: String,
                output_path: String,
                dtype: DTYPE = DTYPE::default(),
                show_progress: bool = true
            )
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

        {
            let elements = granne::AngularVectors::load(dimension, &elements[..]);
            let _: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
                &index[..], &elements
            );
        }

        Hnsw::create_instance(py, index, elements, dimension)
    }

    def search(&self,
               element: Vec<f32>,
               num_elements: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        let elements = granne::AngularVectors::load(*self.dimension(py), self.elements(py));
        let index: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
            self.index(py), &elements
        );

        Ok(index.search(&element.into(), num_elements, max_search))
    }

    def search_batch(&self,
                     elements: Vec<Vec<f32>>,
                     num_elements: usize = DEFAULT_NUM_NEIGHBORS,
                     max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<Vec<(usize, f32)>>>
    {
        let _elements = granne::AngularVectors::load(*self.dimension(py), self.elements(py));
        let index: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
            self.index(py), &_elements
        );

        Ok(elements
           .into_par_iter()
           .map(|element| index.search(&element.into_iter().collect(), num_elements, max_search))
           .collect()
        )
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        let elements = granne::AngularVectors::load(*self.dimension(py), self.elements(py));
        let index: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
            self.index(py), &elements
        );

        Ok(index.get_element(idx).into())
    }

    def __len__(&self) -> PyResult<usize> {
        let elements = granne::AngularVectors::load(*self.dimension(py), self.elements(py));
        let index: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
            self.index(py), &elements
        );

        Ok(index.len())
    }

    def get_neighbors(&self, idx: usize, layer: usize) -> PyResult<Vec<usize>> {
        let elements = granne::AngularVectors::load(*self.dimension(py), self.elements(py));
        let index: granne::Hnsw<granne::AngularVectors, granne::AngularVector> = granne::Hnsw::load(
            self.index(py), &elements
        );

        Ok(index.get_neighbors(idx, layer))
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

enum BuilderType {
    AngularVectorBuilder(granne::HnswBuilder<'static, granne::AngularVectors<'static>, granne::AngularVector<'static>>),
    AngularIntVectorBuilder(
        granne::HnswBuilder<'static, granne::AngularIntVectors<'static>, granne::AngularIntVector<'static>>,
    ),
    MmapAngularVectorBuilder(granne::HnswBuilder<'static, granne::MmapAngularVectors, granne::AngularVector<'static>>),
    MmapAngularIntVectorBuilder(
        granne::HnswBuilder<'static, granne::MmapAngularIntVectors, granne::AngularIntVector<'static>>,
    ),
}

py_class!(class HnswBuilder |py| {
    data builder: RefCell<BuilderType>;

    def __new__(_cls,
                dimension: usize,
                num_layers: usize,
                num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
                max_search: usize = DEFAULT_MAX_SEARCH,
                dtype: DTYPE = DTYPE::default(),
                reinsert_elements: bool = true,
                show_progress: bool = true) -> PyResult<HnswBuilder> {

        let config = granne::Config {
            num_layers: num_layers,
            num_neighbors: num_neighbors,
            max_search: max_search,
            reinsert_elements: reinsert_elements,
            show_progress: show_progress
        };

        match dtype {
            DTYPE::I8 => {
                let builder = granne::HnswBuilder::new(dimension, config);
                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::AngularIntVectorBuilder(builder)))
            },
            DTYPE::F32 => {
                let builder = granne::HnswBuilder::new(dimension, config);
                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::AngularVectorBuilder(builder)))
            }
        }
    }

    @classmethod
    def with_owned_elements(
        _cls,
        dimension: usize,
        num_layers: usize,
        elements_path: &str,
        index_path: Option<String> = None,
        num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
        max_search: usize = DEFAULT_MAX_SEARCH,
        dtype: DTYPE = DTYPE::default(),
        reinsert_elements: bool = true,
        show_progress: bool = true) -> PyResult<HnswBuilder>
    {
        let elements = File::open(elements_path).expect("Could not open elements file");
        let elements = unsafe { Mmap::map(&elements).unwrap() };


        let config = granne::Config {
            num_layers: num_layers,
            num_neighbors: num_neighbors,
            max_search: max_search,
            reinsert_elements: reinsert_elements,
            show_progress: show_progress
        };

        match dtype {
            DTYPE::I8 => {
                let elements = granne::AngularIntVectors::from_vec(dimension, granne::file_io::load(&elements).to_vec());

                let builder = if let Some(index_path) = index_path {
                    let index_file = File::open(index_path).expect("Could not open index file");
                    let mut index_file = BufReader::new(index_file);

                    granne::HnswBuilder::read_index_with_owned_elements(
                        config.clone(),
                        &mut index_file,
                        elements).expect("Could not read index")

                } else {
                    granne::HnswBuilder::with_owned_elements(
                        config.clone(),
                        elements)
                };

                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::AngularIntVectorBuilder(builder)))
            },
            DTYPE::F32 => {
                let elements = granne::AngularVectors::from_vec(dimension, granne::file_io::load(&elements).to_vec());

                let builder = if let Some(index_path) = index_path {
                    let index_file = File::open(index_path).expect("Could not open index file");
                    let mut index_file = BufReader::new(index_file);

                    granne::HnswBuilder::read_index_with_owned_elements(
                        config.clone(),
                        &mut index_file,
                        elements).expect("Could not read index")

                } else {
                    granne::HnswBuilder::with_owned_elements(
                        config.clone(),
                        elements)
                };

                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::AngularVectorBuilder(builder)))
            }
        }
    }

    @classmethod
    def with_mmapped_elements(
        _cls,
        dimension: usize,
        num_layers: usize,
        elements_path: &str,
        index_path: Option<String> = None,
        num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
        max_search: usize = DEFAULT_MAX_SEARCH,
        dtype: DTYPE = DTYPE::default(),
        reinsert_elements: bool = true,
        show_progress: bool = true) -> PyResult<HnswBuilder>
    {
        let config = granne::Config {
            num_layers: num_layers,
            num_neighbors: num_neighbors,
            max_search: max_search,
            reinsert_elements: reinsert_elements,
            show_progress: show_progress
        };

        match dtype {
            DTYPE::I8 => {
                let elements = granne::MmapAngularIntVectors::new(elements_path, dimension);

                let builder = if let Some(index_path) = index_path {
                    let index_file = File::open(index_path).expect("Could not open index file");
                    let mut index_file = BufReader::new(index_file);

                    granne::HnswBuilder::read_index_with_owned_elements(config, &mut index_file, elements).expect("Could not read index")
                } else {
                    granne::HnswBuilder::with_owned_elements(config, elements)
                };

                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::MmapAngularIntVectorBuilder(builder)))
            }
            DTYPE::F32 => {
                let elements = granne::MmapAngularVectors::new(elements_path, dimension);

                let builder = if let Some(index_path) = index_path {
                    let index_file = File::open(index_path).expect("Could not open index file");
                    let mut index_file = BufReader::new(index_file);

                    granne::HnswBuilder::read_index_with_owned_elements(config, &mut index_file, elements).expect("Could not read index")
                } else {
                    granne::HnswBuilder::with_owned_elements(config, elements)
                };

                return HnswBuilder::create_instance(py, RefCell::new(BuilderType::MmapAngularVectorBuilder(builder)))
            }
        }
    }


    def add(&self, element: Vec<f32>) -> PyResult<PyObject> {
        match *self.builder(py).borrow_mut() {
            BuilderType::AngularVectorBuilder(ref mut builder) => {
                let mut elements = granne::AngularVectors::new(0);
                elements.push(&element.into());

                builder.add(elements);
            },
            BuilderType::AngularIntVectorBuilder(ref mut builder) => {
                let mut elements = granne::AngularIntVectors::new(0);
                elements.push(&element.into());

                builder.add(elements);
            },
            _ => panic!("Cannot add element to mmapped builder")
        }

        Ok(py.None())
    }

    def __len__(&self) -> PyResult<usize> {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => Ok(builder.len()),
            BuilderType::AngularIntVectorBuilder(ref builder) => Ok(builder.len()),
            BuilderType::MmapAngularVectorBuilder(ref builder) => Ok(builder.len()),
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => Ok(builder.len()),
        }
    }

    def indexed_elements(&self) -> PyResult<usize> {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => Ok(builder.indexed_elements()),
            BuilderType::AngularIntVectorBuilder(ref builder) => Ok(builder.indexed_elements()),
            BuilderType::MmapAngularVectorBuilder(ref builder) => Ok(builder.indexed_elements()),
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => Ok(builder.indexed_elements()),
        }
    }

    def __getitem__(&self, idx: usize) -> PyResult<Vec<f32>> {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => Ok(builder.get_index().get_element(idx).into()),
            BuilderType::AngularIntVectorBuilder(ref builder) => Ok(builder.get_index().get_element(idx).into()),
            BuilderType::MmapAngularVectorBuilder(ref builder) => Ok(builder.get_index().get_element(idx).into()),
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => Ok(builder.get_index().get_element(idx).into()),
        }
    }


    def build_index(&self, num_elements: usize = <usize>::max_value()) -> PyResult<PyObject> {
        match *self.builder(py).borrow_mut() {
            BuilderType::AngularVectorBuilder(ref mut builder) => {
                if num_elements == <usize>::max_value() {
                    builder.build_index();
                } else {
                    builder.build_index_part(num_elements);
                }
            },
            BuilderType::AngularIntVectorBuilder(ref mut builder) => {
                if num_elements == <usize>::max_value() {
                    builder.build_index();
                } else {
                    builder.build_index_part(num_elements);
                }
            },
            BuilderType::MmapAngularVectorBuilder(ref mut builder) => {
                if num_elements == <usize>::max_value() {
                    builder.build_index();
                } else {
                    builder.build_index_part(num_elements);
                }
            },
            BuilderType::MmapAngularIntVectorBuilder(ref mut builder) => {
                if num_elements == <usize>::max_value() {
                    builder.build_index();
                } else {
                    builder.build_index_part(num_elements);
                }
            },
        }

        Ok(py.None())
    }


    def save_index(&self, path: &str, compress: bool = false) -> PyResult<PyObject> {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => builder.save_index_to_disk(path, compress),
            BuilderType::AngularIntVectorBuilder(ref builder) => builder.save_index_to_disk(path, compress),
            BuilderType::MmapAngularVectorBuilder(ref builder) => builder.save_index_to_disk(path, compress),
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => builder.save_index_to_disk(path, compress),
        }.expect("Could not save index to disk");

        Ok(py.None())
    }

    def save_elements(&self, path: &str) -> PyResult<PyObject> {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => builder.save_elements_to_disk(path),
            BuilderType::AngularIntVectorBuilder(ref builder) => builder.save_elements_to_disk(path),
            BuilderType::MmapAngularVectorBuilder(ref builder) => builder.save_elements_to_disk(path),
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => builder.save_elements_to_disk(path),
        }.expect("Could not save elements to disk");

        Ok(py.None())
    }

    def search(&self, element: Vec<f32>,
               num_neighbors: usize = DEFAULT_NUM_NEIGHBORS,
               max_search: usize = DEFAULT_MAX_SEARCH) -> PyResult<Vec<(usize, f32)>>
    {
        match *self.builder(py).borrow() {
            BuilderType::AngularVectorBuilder(ref builder) => {
                let index = builder.get_index();
                Ok(index.search(&element.into(), num_neighbors, max_search))
            },
            BuilderType::AngularIntVectorBuilder(ref builder) => {
                let index = builder.get_index();
                Ok(index.search(&element.into(), num_neighbors, max_search))
            },
            BuilderType::MmapAngularVectorBuilder(ref builder) => {
                let index = builder.get_index();
                Ok(index.search(&element.into(), num_neighbors, max_search))
            },
            BuilderType::MmapAngularIntVectorBuilder(ref builder) => {
                let index = builder.get_index();
                Ok(index.search(&element.into(), num_neighbors, max_search))
            },
        }
    }

});
