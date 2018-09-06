use super::{Hnsw, HnswBuilder, Config, ShardedHnsw, At, Writeable};
use types::{ComparableTo, Dense, AngularVector};
use mmap::MmapSlice;
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
    fn build_index(self: &mut Self);
    fn build_index_part(self: &mut Self, num_elements: usize);
    fn len(self: &Self) -> usize;
    fn indexed_elements(self: &Self) -> usize;
    fn save_index_to_disk(self: &Self, path: &str) -> io::Result<()>;
    fn save_elements_to_disk(self: &Self, path: &str) -> io::Result<()>;
    fn get_index<'a>(self: &'a Self) -> Box<SearchIndex + 'a>;
}


impl<'a, Elements, Element> SearchIndex for Hnsw<'a, Elements, Element>
    where Elements: 'a + At<Output=Element> + ?Sized,
          Element: 'a + ComparableTo<Element> + Dense<f32> + FromIterator<f32>
{
    fn search(self: &Self,
              element: Vec<f32>,
              num_elements: usize,
              max_search: usize) -> Vec<(usize, f32)> {
        self.search(&element.into_iter().collect(), num_elements, max_search)
    }

    fn get_element(self: &Self, idx: usize) -> Vec<f32> {
        self.get_element(idx).as_slice().to_vec()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a, Element, Elements> IndexBuilder for HnswBuilder<'a, Elements, Element>
    where Element: 'a + ComparableTo<Element> + Dense<f32> + FromIterator<f32> + Clone + Send + Sync,
          Elements: 'a + At<Output=Element> + ToOwned + Writeable + Send + Sync + ?Sized
{
    fn build_index(self: &mut Self) {
        self.build_index();
    }

    fn build_index_part(self: &mut Self, num_elements: usize) {
        self.build_index_part(num_elements);
    }

    fn len(self: &Self) -> usize {
        self.len()
    }

    fn indexed_elements(self: &Self) -> usize {
        self.indexed_elements()
    }

    fn save_index_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.save_index_to_disk(path)
    }

    fn save_elements_to_disk(self: &Self, path: &str) -> io::Result<()> {
        self.save_elements_to_disk(path)
    }

    fn get_index<'b>(self: &'b Self) -> Box<SearchIndex + 'b> {
        Box::new(self.get_index())
    }
}



///
/// The macro match_dimension expands to a big match statement where each match arm corresponds to
/// one valid dimension (currently specified inside the macro itself). Note:
/// - The matched expression needs to be wrapped in parentheses.
/// - All match arms need to be blocks {...} and must return compatible types, i.e., it is not possible
/// to return Hnsw objects with different element types.
///
/// Example usage for getting a boxed SearchIndex where the underlying Hnsw object can have varying
/// dimension:
///
/// match_dimension!((dim) {
///     DIM => {
///         Box::new(
///             Hnsw::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::load(
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
///                 Hnsw::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::load(
///                     index, file_io::load(elements)
///                 )
///             )
///         }
///     },
///     3 => {
///         const DIM: usize = 3;
///         {
///             Box::new(
///                 Hnsw::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::load(
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
///                 Hnsw::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::load(
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


/// Returns an Hnsw index for elements of type AngularVector<[f32; dim]> wrapped in a Box
pub fn boxed_index<'a>(index: &'a [u8],
                       elements: &'a [u8],
                       dim: usize) -> Box<SearchIndex + 'a>
{
    match_dimension!((dim) {
        DIM => {
            Box::new(
                Hnsw::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::load(
                    index, file_io::load(elements)
                )
            )
        }
    })
}


pub fn boxed_sharded_index<'a>(shards: &[(&'a [u8], &'a [u8])],
                               dim: usize) -> Box<SearchIndex + 'a>
{
    match_dimension!((dim) {
        DIM => {
            type Element = AngularVector<[f32; DIM]>;
            let shards: Vec<_> = (shards).iter().map(|&(idx, elements)| {
                (idx, file_io::load::<Element>(elements))
            }).collect();

            Box::new(
                ShardedHnsw::<[Element], Element>::new(&shards)
            )
        }
    })
}


/// Returns an empty HnswBuilder for elements of type AngularVector<[f32; dim]> wrapped in a Box
pub fn boxed_builder(dim: usize,
                     config: Config) -> Box<IndexBuilder + Send>
{
    match_dimension!((dim) {
        DIM => {
            Box::new(
                HnswBuilder::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::new(config)
            )
        }
    })
}


/// Returns an HnswBuilder for elements of type AngularVector<[f32; dim]> wrapped in a Box with its own clone
/// of the elements
pub fn boxed_owning_builder(dim: usize,
                            config: Config,
                            elements: &[u8],
                            existing_index: Option<io::BufReader<::std::fs::File>>) -> Box<IndexBuilder + Send>
{
    if let Some(mut existing_index) = existing_index {
        match_dimension!((dim) {
            DIM => {
                let elements = file_io::load::<[f32; DIM]>(elements).iter().map(|&element| element.into()).collect();

                Box::new(
                    HnswBuilder::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::read_index_with_owned_elements(
                        config, &mut existing_index, elements).expect("Could not read index")
                )
            }
        })
    } else {
        match_dimension!((dim) {
            DIM => {
                let elements = file_io::load::<[f32; DIM]>(elements).iter().map(|&element| element.into()).collect();

                let mut builder = HnswBuilder::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::new(config);
                builder.add(elements);

                Box::new(builder)
            }
        })
    }
}


/// Returns an HnswBuilder for elements of type AngularVector<[f32; dim]> wrapped in a Box with borrowed elements
pub fn boxed_borrowing_builder<'a>(dim: usize,
                                   config: Config,
                                   elements: &'a [u8],
                                   existing_index: Option<io::BufReader<::std::fs::File>>) -> Box<IndexBuilder + Send + 'a>
{
    if let Some(mut existing_index) = existing_index {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::read_index_with_owned_elements(
                        config, &mut existing_index, file_io::load(elements).to_vec()).expect("Could not read index")
                )
            }
        })
    } else {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<[AngularVector<[f32; DIM]>], AngularVector<[f32; DIM]>>::with_borrowed_elements(
                        config, file_io::load(elements))
                )
            }
        })
    }
}


/// Returns an HnswBuilder for elements of type AngularVector<[f32; dim]> wrapped in a Box with mmapped elements
pub fn boxed_mmap_builder<'a>(dim: usize,
                              config: Config,
                              elements_path: &str,
                              existing_index: Option<io::BufReader<::std::fs::File>>) -> Box<IndexBuilder + Send>
{
    if let Some(mut existing_index) = existing_index {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<MmapSlice<AngularVector<[f32; DIM]>>, AngularVector<[f32; DIM]>>::read_index_with_owned_elements(
                        config, &mut existing_index, MmapSlice::new(elements_path)).unwrap()
                )
            }
        })
    } else {
        match_dimension!((dim) {
            DIM => {
                Box::new(
                    HnswBuilder::<MmapSlice<AngularVector<[f32; DIM]>>, AngularVector<[f32; DIM]>>::with_owned_elements(
                        config, MmapSlice::new(elements_path))
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
