use crossbeam;
use ordered_float::NotNaN;

use super::{At, Hnsw};

use crate::types::ComparableTo;

pub struct ShardedHnsw<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + Sync + ?Sized,
    Element: 'a + ComparableTo<Element> + Sync,
{
    indexes: Vec<(Hnsw<'a, Elements, Element>, usize)>,
}

impl<'a, Elements, Element> ShardedHnsw<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + Sync + ?Sized,
    Element: 'a + ComparableTo<Element> + Sync,
{
    pub fn new(shards: &[(&'a [u8], &'a Elements)]) -> Self {
        let mut cumulative_offset = 0;
        let indexes = shards
            .iter()
            .map(|&(index, elements)| {
                let index = Hnsw::<Elements, Element>::load(index, elements);
                let id_offset = cumulative_offset;
                cumulative_offset += index.len();
                (index, id_offset)
            })
            .collect();

        Self { indexes: indexes }
    }

    pub fn search(&self, element: &Element, num_neighbors: usize, max_search: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<_> = crossbeam::scope(|scope| {
            let children: Vec<_> = (0..self.indexes.len())
                .map(|i| {
                    scope.spawn(move || {
                        let (ref index, id_offset) = self.indexes[i];

                        let mut results: Vec<(usize, f32)> = index.search(element, num_neighbors, max_search);
                        results.iter_mut().for_each(|&mut (ref mut id, _)| *id += id_offset);
                        results
                    })
                })
                .collect();

            children.into_iter().flat_map(|child| child.join()).collect()
        });

        // sort and pick top num_neighbor elements
        // sort is based on tim sort which is optimized for cases with two or more sorted sequences concatenated
        results.sort_by_key(|&(_, d)| NotNaN::new(d).unwrap());
        results.truncate(num_neighbors);

        results
    }

    pub fn sequential_search(&self, element: &Element, num_neighbors: usize, max_search: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<_> = self
            .indexes
            .iter()
            .flat_map(|&(ref index, id_offset)| {
                let mut results = index.search(element, num_neighbors, max_search);
                results.iter_mut().for_each(|&mut (ref mut id, _)| *id += id_offset);
                results
            })
            .collect();

        // sort and pick top num_neighbor elements
        // sort is based on tim sort which is optimized for cases with two or more sorted sequences concatenated
        results.sort_by_key(|&(_, d)| NotNaN::new(d).unwrap());
        results.resize(num_neighbors, (0, 0.0));

        results
    }

    pub fn get_element(self: &Self, idx: usize) -> Element {
        let mut cumulative_offset = 0;
        for &(ref index, _) in self.indexes.iter() {
            if idx < cumulative_offset + index.len() {
                return index.get_element(idx - cumulative_offset);
            } else {
                cumulative_offset += index.len();
            }
        }

        panic!("Index out of bounds");
    }

    pub fn len(self: &Self) -> usize {
        self.indexes.iter().map(|&(ref index, _)| index.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::*;
    use crate::types::example::*;
    use crate::types::AngularVector;
    use std::io::{Seek, SeekFrom};
    use tempfile;

    type ElementType = AngularVector<'static>;

    fn get_shards(num_shards: usize, num_elements: usize) -> Vec<(Vec<u8>, Vec<ElementType>)> {
        assert!(num_elements % num_shards == 0);
        let elements: Vec<_> = (0..num_elements).map(|_| random_dense_element(25)).collect();
        let shards: Vec<_> = elements
            .chunks(num_elements / num_shards)
            .map(|chunk| {
                let mut builder = HnswBuilder::<[ElementType], ElementType>::with_owned_elements(
                    Config {
                        num_layers: 5,
                        max_search: 50,
                        show_progress: false,
                        num_neighbors: 20,
                    },
                    chunk.to_vec(),
                );

                builder.build_index();
                let mut file: File = tempfile::tempfile().unwrap();
                builder.write(&mut file).unwrap();
                file.seek(SeekFrom::Start(0)).unwrap();
                let mut data = Vec::new();
                file.read_to_end(&mut data).unwrap();

                (data, chunk.to_vec())
            })
            .collect();

        shards
    }

    #[test]
    fn sharded_vs_sequential() {
        let shards = get_shards(10, 1000);

        let elements: Vec<ElementType> = shards
            .iter()
            .flat_map(|&(_, ref chunk)| chunk.clone().into_iter())
            .collect();

        let sharded_index = ShardedHnsw::<[ElementType], ElementType>::new(
            &shards
                .iter()
                .map(|&(ref index, ref elements)| (index.as_slice(), elements.as_slice()))
                .collect::<Vec<_>>(),
        );

        let max_search = 25;
        let num_neighbors = 7;

        for i in vec![0, 10, 100, 250] {
            let result = sharded_index.search(&elements[i], num_neighbors, max_search);
            let result_seq = sharded_index.search(&elements[i], num_neighbors, max_search);

            assert_eq!(result.len(), result_seq.len());

            for ((a, _), (b, _)) in result.into_iter().zip(result_seq.into_iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn offsets() {
        let shards = get_shards(10, 1000);
        let elements: Vec<ElementType> = shards
            .iter()
            .flat_map(|&(_, ref chunk)| chunk.clone().into_iter())
            .collect();

        let sharded_index = ShardedHnsw::<[ElementType], ElementType>::new(
            &shards
                .iter()
                .map(|&(ref index, ref elements)| (index.as_slice(), elements.as_slice()))
                .collect::<Vec<_>>(),
        );

        assert_eq!(elements.len(), sharded_index.len());

        for i in vec![0, 100, 250, 999] {
            let results = sharded_index.search(&elements[i], 10, 20);

            for (id, _) in results {
                assert!(id < elements.len());
            }
        }
    }
}
