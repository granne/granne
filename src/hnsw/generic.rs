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
    fn save_index_to_disk(self: &Self, path: &str) -> io::Result<()>;
    fn save_elements_to_disk(self: &Self, path: &str) -> io::Result<()>;
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

impl<'a, T: 'a + ComparableTo<T> + Dense<f32> + FromIterator<f32> + Clone + Send + Sync> IndexBuilder for HnswBuilder<'a, T>
{
    fn add(self: &mut Self, element: Vec<f32>) {
        self.add(vec![element.into_iter().collect()]);
    }

    fn build(self: &mut Self) {
        self.build_index();
    }

    fn save_elements_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.save_elements_to_disk(path)
    }

    fn save_index_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.save_index_to_disk(path)
    }

    fn get_index<'b>(self: &'b Self) -> Box<SearchIndex + 'b> {
        Box::new(self.get_index())
    }
}


///
/// The macro match_dimension expands to a big match statement where each match arm corresponds to
/// one valid dimension (currently specified inside the macro itself). Note:
/// - The matched expression needs to be wrapped in parentheses.
/// - All match arms (need to be blocks {...} and must return compatible types, i.e., it is not possible
/// to return Hnsw objects with different element types.
///
/// Example usage for getting a boxed SearchIndex where the underlying Hnsw object can have varying
/// dimension:
///
/// match_dimension!((dim) {
///     DIM => {
///         Box::new(
///             Hnsw::<AngularVector<[f32; DIM]>, AngularVector<[f32; DIM]>>::load(
///                 index, file_io::load(elements)
///             )
///         )
///     }
/// })
///
/// The example will expand to something similar to this:
///
/// match dim {
///     2 => {
///         const DIM: usize = 2;
///         {
///             Box::new(
///                 Hnsw::<AngularVector<[f32; DIM]>, AngularVector<[f32; DIM]>>::load(
///                     index, file_io::load(elements)
///                 )
///             )
///         }
///     },
///     3 => {
///         const DIM: usize = 3;
///         {
///             Box::new(
///                 Hnsw::<AngularVector<[f32; DIM]>, AngularVector<[f32; DIM]>>::load(
///                     index, file_io::load(elements)
///                 )
///             )
///         }
///     },
///     ...
///
///     300 => {
///         const DIM: usize = 300;
///         {
///             Box::new(
///                 Hnsw::<AngularVector<[f32; DIM]>, AngularVector<[f32; DIM]>>::load(
///                     index, file_io::load(elements)
///                 )
///             )
///         }
///     },
///     _ => panic!("Unsupported dimension")
/// }
///

macro_rules! match_dimension {
    (($dim:expr) { $DIM:ident => $body:block }) => {
        match_dimension!(
            $body, $dim, $DIM,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30,
            32, 50, 60, 64, 96, 100, 128, 200, 256, 300)
    };
    ($body:block, $dim:expr, $DIM:ident, $($dims:expr),+) => {
        {
            match $dim {
                $($dims => {
                    const $DIM: usize = $dims;
                    $body
                },)+
                    _ => panic!("Unsupported dimension"),
            }
        }
    };
}

pub fn boxed_index<'a>(index: &'a [u8],
                       elements: &'a [u8],
                       dim: usize) -> Box<SearchIndex + 'a>
{
    match_dimension!((dim) {
        DIM => {
            Box::new(
                Hnsw::<AngularVector<[f32; DIM]>, AngularVector<[f32; DIM]>>::load(
                    index, file_io::load(elements)
                )
            )
        }
    })
}


pub fn boxed_index_builder<'a>(dim: usize,
                               config: Config,
                               elements: Option<&'a [u8]>) -> Box<IndexBuilder + Send + 'a>
{
    if let Some(elements) = elements {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<AngularVector<[f32; DIM]>>::with_elements(
                        config, file_io::load(elements))
                )
            }
        })
    } else {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<AngularVector<[f32; DIM]>>::new(config)
                )
            }
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn match_dimension() {
        for &i in &[2,3,4,5,10,15,25,30,64,256,300] {
            assert_eq!(2 * i, match_dimension!((i) { DIMENSION =>  { 2 * DIMENSION } }));
        }
    }
}
