use super::{Hnsw, HnswBuilder, Config};
use types::{ComparableTo, Dense, AngularVector};
use file_io;

use std::iter::{FromIterator};
use std::io;

pub trait SearchIndex {
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)>;

    fn get_element(self: &Self, idx: usize) -> Vec<f32>;

    fn len(self: &Self) -> usize;
}

pub trait IndexBuilder {
    fn add(self: &mut Self, element: Vec<f32>);
    fn build(self: &mut Self);
    fn save_to_disk(self: &Self, path: &str) -> io::Result<()>;
    fn get_index<'a>(self: &'a Self) -> Box<SearchIndex + 'a>;
}

impl<'a, T: 'a + ComparableTo<E> + Dense<f32>, E: FromIterator<f32>> SearchIndex for Hnsw<'a, T, E> {
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)> {
        self.search(&element.into_iter().collect(), num_elements, max_search)
    }

    fn get_element(self: &Self, idx: usize) -> Vec<f32> {
        self.elements[idx].as_slice().to_vec()
    }

    fn len(self: &Self) -> usize {
        self.elements.len()
    }
}

impl<T: ComparableTo<T> + Dense<f32> + FromIterator<f32> + Clone + Send + Sync> IndexBuilder for HnswBuilder<T> 
{
    fn add(self: &mut Self, element: Vec<f32>) {
        self.add(vec![element.into_iter().collect()]);
    }

    fn build(self: &mut Self) {
        self.build_index();
    }

    fn save_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.save_to_disk(path)
    }

    fn get_index<'a>(self: &'a Self) -> Box<SearchIndex + 'a> {
        Box::new(self.get_index())
    }
}

macro_rules! match_dimension_and_get_index {
    ($index:expr, $elements:expr, $dim:expr, $($dims:expr),+) => {
        {
            match $dim {
                $($dims => {
                    Box::new(
                        Hnsw::<
                            AngularVector<[f32; $dims]>, 
                            AngularVector<[f32; $dims]>
                        >::load(
                            $index,
                            file_io::load($elements)
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

pub fn boxed_index<'a>(index: &'a [u8], 
                       elements: &'a [u8], 
                       dim: usize) -> Box<SearchIndex + 'a> 
{
    boxed_index!(index, elements, dim)
}


macro_rules! match_dimension_and_get_index_builder {
    ($config:expr, $dim:expr, $($dims:expr),+) => {
        {
            match $dim {
                $($dims => {
                    Box::new(
                        HnswBuilder::<
                            AngularVector<[f32; $dims]>
                        >::new($config)
                    )
                },)+
                _ => panic!("Unsupported dimension"),
            }
        }
    };
}

macro_rules! boxed_index_builder {
    ($config:expr, $dim:expr) => {
        match_dimension_and_get_index_builder!(
            $config, $dim,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30,
            32, 50, 60, 64, 96, 100, 128, 200, 256, 300)
    };
}

pub fn boxed_index_builder(config: Config, 
                           dim: usize) -> Box<IndexBuilder + Send> 
{
    boxed_index_builder!(config, dim)
}


