#![allow(missing_docs)]

use super::*;

use parking_lot::{self, RwLock};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

use crate::io::Writeable;
use crate::slice_vector::FixedWidthSliceVector;

mod rw_lock_slice_vector;

/// A version of GranneBuilder that can build and search concurrently.
pub struct RwGranneBuilder<Elements: ExtendableElementContainer + Writeable> {
    layers: RwLock<(
        rw_lock_slice_vector::RwLockSliceVector,
        Vec<FixedWidthSliceVector<'static, NeighborId>>,
    )>,
    elements: RwLock<Elements>,
    config: BuildConfig,
    max_elements: usize,
    pool: rayon::ThreadPool,
    // only used to prevent writes from happening while writing to disk
    write_lock: RwLock<()>,
}

impl<Elements> RwGranneBuilder<Elements>
where
    Elements: ExtendableElementContainer + Writeable + Sync + Send + Clone,
{
    pub fn new(builder: GranneBuilder<Elements>, max_elements: usize, num_threads: usize) -> Self {
        let mut builder = builder;
        builder.config.expected_num_elements = Some(max_elements);

        builder.build();

        let mut current_layer = builder
            .layers
            .pop()
            .unwrap_or_else(|| FixedWidthSliceVector::with_width(builder.config.num_neighbors));

        let num_elements_in_layer = cmp::max(
            current_layer.len(),
            compute_num_elements_in_layer(max_elements, builder.config.layer_multiplier, builder.layers.len()),
        );

        current_layer.resize(num_elements_in_layer, UNUSED);

        Self {
            layers: RwLock::new((current_layer.into(), builder.layers)),
            elements: RwLock::new(builder.elements),
            config: builder.config,
            max_elements,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Could not create threadpool"),
            write_lock: RwLock::new(()),
        }
    }

    pub fn save_index_and_elements_to_disk(self: &Self, index_path: &str, elements_path: &str) -> std::io::Result<()> {
        let mut elements_file = File::create(elements_path)?;
        let mut index_file = File::create(index_path)?;

        self.write(&mut index_file, &mut elements_file)
    }

    pub fn write(self: &Self, index_file: impl Write + Seek, elements_file: impl Write) -> std::io::Result<()> {
        // prevent any writes from happening while saving to disk
        let _guard = self.write_lock.write();

        let elements = self.elements.read();
        let (ref current_layer, ref layers) = *self.layers.read();

        let mut elements_file = BufWriter::new(elements_file);
        elements.write(&mut elements_file)?;

        // this is safe because we have a write lock on self.write_lock, so no writes
        // can happen
        let last_layer: &[NeighborId] = unsafe { current_layer.as_owner() };
        let last_layer = FixedWidthSliceVector::with_data(last_layer, self.config.num_neighbors);

        let layers = if elements.len() > 0 {
            let mut layers: Vec<_> = layers.iter().map(|layer| layer.borrow()).collect();

            layers.push(last_layer.subslice(0, elements.len()));

            layers
        } else {
            vec![]
        };

        let layers: Layers = Layers::FixWidth(layers.iter().map(|layer| layer.borrow()).collect());
        io::write_index(&layers, index_file)
    }

    pub fn insert(self: &Self, element: Elements::InternalElement) -> Option<usize> {
        self.insert_batch(vec![element]).pop()
    }

    pub fn insert_batch(self: &Self, mut elements_to_insert: Vec<Elements::InternalElement>) -> Vec<usize> {
        if self.elements.read().len() >= self.max_elements {
            return vec![];
        }

        // signal that writes will happen
        let _guard = self.write_lock.read();

        let (mut inserted, remaining) = {
            // write locks are needed to append elements and potentially append a new layer
            let mut elements = self.elements.write();
            let mut layers = self.layers.write();
            let (ref mut current_layer, ref mut prev_layers) = *layers;

            // check if it is time to create a new layer
            if elements.len() >= current_layer.len() {
                *current_layer = {
                    // temporarily replace with empty slice vector
                    let current_layer: FixedWidthSliceVector<_> =
                        std::mem::replace(current_layer, FixedWidthSliceVector::with_width(1).into()).into();
                    let mut new_layer = current_layer.clone();
                    prev_layers.push(current_layer);

                    let num_elements_in_layer = compute_num_elements_in_layer(
                        self.max_elements,
                        self.config.layer_multiplier,
                        prev_layers.len(),
                    );

                    new_layer.resize(num_elements_in_layer, UNUSED);
                    new_layer.into()
                };
            }

            assert!(current_layer.len() >= elements.len());

            // insert elements that fit in the current layer (the remaining will be inserted in a
            // recursive call)
            let num_to_insert = std::cmp::min(elements_to_insert.len(), current_layer.len() - elements.len());
            let ids: Vec<usize> = (elements.len()..).take(num_to_insert).collect();

            let remaining = elements_to_insert.split_off(num_to_insert);

            for element in elements_to_insert {
                elements.push(element);
            }

            // downgrade to read locks before inserting into graph
            let (ref current_layer, ref layers) = *parking_lot::RwLockWriteGuard::downgrade(layers);
            let elements = parking_lot::RwLockWriteGuard::downgrade(elements);

            let index = Granne::from_parts(
                Layers::FixWidth(layers.iter().map(|layer| layer.borrow()).collect()),
                &*elements,
            );

            if self.pool.current_num_threads() > 1 {
                self.pool.install(|| {
                    ids.par_iter().for_each(|id| {
                        GranneBuilder::index_element(&self.config, &*elements, &index, current_layer.as_slice(), *id)
                    })
                });
            } else {
                ids.iter().for_each(|id| {
                    GranneBuilder::index_element(&self.config, &*elements, &index, current_layer.as_slice(), *id)
                })
            }

            // locks go out of scope and are released before potentially inserting the remaining
            // elements
            (ids, remaining)
        };

        if !remaining.is_empty() {
            let ids = self.insert_batch(remaining);
            inserted.extend(ids.into_iter());
        }

        inserted
    }

    pub fn search(
        self: &Self,
        element: &Elements::Element,
        max_search: usize,
        num_neighbors: usize,
    ) -> Vec<(usize, f32)> {
        let elements = self.elements.read();
        let (ref current_layer, ref layers) = *self.layers.read();

        let index = Granne::from_parts(
            Layers::FixWidth(layers.iter().map(|layer| layer.borrow()).collect()),
            &*elements,
        );

        if let Some((entrypoint, _)) = index.search(&element, 1, 1).first() {
            super::search_for_neighbors(current_layer.as_slice(), *entrypoint, &*elements, element, max_search)
                .into_iter()
                .take(num_neighbors)
                .map(|(i, d)| (i, d.into_inner()))
                .collect()
        } else {
            vec![]
        }
    }

    pub fn get_elements(self: &Self) -> parking_lot::RwLockReadGuard<Elements> {
        self.elements.read()
    }

    pub fn get_element(self: &Self, idx: usize) -> Elements::Element {
        self.elements.read().get(idx)
    }

    pub fn len(self: &Self) -> usize {
        self.elements.read().len()
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    #[test]
    fn insert_in_parallel() {
        let builder = GranneBuilder::new(
            BuildConfig::default().max_search(50).reinsert_elements(false),
            test_helper::random_sum_embeddings(5, 505, 0),
        );

        let builder = RwGranneBuilder::new(builder, 5000, 1);

        for x in 0..100 {
            builder.insert(vec![x]).expect("Did not insert element");
        }

        rayon::join(
            || {
                for x in 0..500 {
                    builder
                        .insert(vec![x, x + 1, x + 2, x + 3, x + 4])
                        .expect("Did not insert element");
                }
            },
            || {
                (0..1000usize).into_par_iter().for_each(|_| {
                    builder.search(&test_helper::random_vector(5), 5, 50);
                });
            },
        );
    }

    #[test]
    fn insert_batch() {
        let max_elements = 1500;
        let num_threads = 4;

        let builder = GranneBuilder::new(
            BuildConfig::default()
                .layer_multiplier(5.0)
                .num_neighbors(10)
                .max_search(20)
                .reinsert_elements(false),
            test_helper::random_sum_embeddings(5, 2000, 0),
        );

        let builder = RwGranneBuilder::new(builder, max_elements, num_threads);

        let mut queries = Vec::new();
        for i in 0..(max_elements + 100) {
            queries.push(vec![i]);
        }

        assert_eq!(
            (0..100).collect::<Vec<usize>>(),
            builder.insert_batch(queries[..100].to_vec())
        );
        assert_eq!(
            (100..120).collect::<Vec<usize>>(),
            builder.insert_batch(queries[100..120].to_vec())
        );
        assert_eq!(Some(120), builder.insert(queries[120].clone()));
        assert_eq!(
            (121..1421).collect::<Vec<usize>>(),
            builder.insert_batch(queries[121..1421].to_vec())
        );
        assert_eq!(
            (1421..1500).collect::<Vec<usize>>(),
            builder.insert_batch(queries[1421..].to_vec())
        );

        for i in 0..max_elements {
            assert_eq!(i, builder.search(&builder.get_element(i), 20, 1)[0].0);
        }
    }

    #[test]
    fn layer_counts() {
        const DIM: usize = 2;
        let num_threads = 1;

        for layer_multiplier in vec![10.0, 15.0, 25.0] {
            let build_config = BuildConfig::default()
                .layer_multiplier(layer_multiplier)
                .max_search(50)
                .reinsert_elements(false);

            for max_elements in vec![13, 66, 199, 719] {
                let builder = GranneBuilder::new(build_config.clone(), crate::elements::angular::Vectors::new());

                let rw_builder = RwGranneBuilder::new(builder, max_elements, num_threads);

                for _ in 0..max_elements {
                    rw_builder.insert(test_helper::random_vector(DIM)).unwrap();
                }

                let elements = rw_builder.elements.read().clone();
                let mut builder = GranneBuilder::new(build_config.clone(), elements);

                builder.build();

                assert_eq!(builder.layers.len(), rw_builder.layers.read().1.len() + 1);

                for (i, layer) in rw_builder.layers.read().1.iter().enumerate() {
                    assert_eq!(builder.layers[i].len(), layer.len());
                }

                assert_eq!(
                    builder.layers.last().unwrap().len(),
                    rw_builder.layers.read().0.as_slice().len()
                );
            }
        }
    }

    #[test]
    fn search_empty() {
        let builder = GranneBuilder::new(
            BuildConfig::default().max_search(50).reinsert_elements(false),
            test_helper::random_sum_embeddings(10, 100, 0),
        );

        let builder = RwGranneBuilder::new(builder, 5000, 1);

        builder.search(&test_helper::random_vector(5), 50, 5);
    }

    #[test]
    fn search_with_one_element() {
        let builder = GranneBuilder::new(
            BuildConfig::default().max_search(50).reinsert_elements(false),
            test_helper::random_sum_embeddings(10, 100, 0),
        );

        let builder = RwGranneBuilder::new(builder, 5000, 1);

        builder.insert(vec![1, 2, 3]);
        builder.search(&test_helper::random_vector(5), 50, 5);
    }
    /*
        #[test]
        fn save_and_read() {
            let config = Config {
                num_layers: 4,
                num_neighbors: 20,
                max_search: 50,
                reinsert_elements: true,
                show_progress: false,
            };

            let dim = 5;
            let max_elements = 500;
            let num_threads = 1;
            let num_embeddings = 10000;

            let word_embeddings = crate::query_embeddings::example::get_random_word_embeddings(dim, num_embeddings);
            let mut embeddings_file: Vec<u8> = Vec::new();
            word_embeddings.write(&mut embeddings_file).unwrap();

            let builder = RwGranneBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
                config.clone(),
                QueryEmbeddings::new(word_embeddings.clone()),
                max_elements,
                num_threads,
            );

            let elements_per_batch = max_elements / 5;

            for i in 0..5 {
                let mut index_file: File = tempfile::tempfile().unwrap();
                let mut elements_file: Vec<u8> = Vec::new();
                builder.write(&mut index_file, &mut elements_file, false).unwrap();

                index_file.seek(SeekFrom::Start(0)).unwrap();

                let elements = crate::query_embeddings::QueryEmbeddings::load(dim, &embeddings_file, &elements_file);

                let read_builder =
                    GranneBuilder::<crate::query_embeddings::QueryEmbeddings, AngularVector<'static>>::read_index_with_borrowed_elements(config.clone(), &mut index_file, &elements).unwrap();

                assert_eq!(builder.len(), read_builder.indexed_elements());

                let read_builder = RwGranneBuilder::from_hnsw_builder(read_builder, max_elements, num_threads);

                assert_eq!(read_builder.len(), builder.len());
                if !read_builder.is_empty() {
                    for j in 0..10 {
                        let element = elements.get_embedding_for_query(&[(i * 100 + j) % num_embeddings]);
                        assert_eq!(
                            builder.search(&element, 100, 10),
                            read_builder.search(&element, 100, 10)
                        );
                    }
                }

                // insert some queries for the next iteration
                let queries: Vec<Vec<usize>> = (0..elements_per_batch)
                    .map(|j| vec![j % word_embeddings.len()])
                    .collect();
                assert_eq!(queries.len(), builder.insert_batch(queries).len());
            }
        }
    */
}
