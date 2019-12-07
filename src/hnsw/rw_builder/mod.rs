use super::*;

use parking_lot::{self, RwLock};
use std::io::Seek;

use slice_vector::FixedWidthSliceVector;

mod rw_lock_slice_vector;

pub struct RwHnswBuilder<Elements, Element, QElement>
where
    Elements: At<Output = Element> + Appendable<Element = QElement> + Writeable + Sync + Send + ToOwned,
    Element: ComparableTo<Element> + Sync + Send,
{
    layers: RwLock<(
        rw_lock_slice_vector::RwLockSliceVector,
        Vec<FixedWidthSliceVector<'static, NeighborId>>,
    )>,
    elements: RwLock<Elements>,
    config: Config,
    max_elements: usize,
    pool: rayon::ThreadPool,
    // only used to prevent writes from happening while writing to disk
    write_lock: RwLock<()>,
}

impl<Elements, Element, QElement> RwHnswBuilder<Elements, Element, QElement>
where
    Elements: At<Output = Element> + Appendable<Element = QElement> + Writeable + Sync + Send + Clone,
    Element: ComparableTo<Element> + Sync + Send,
{
    pub fn new(config: Config, elements: Elements, max_elements: usize, num_threads: usize) -> Self {
        let mut builder = HnswBuilder::with_owned_elements(config, elements);

        builder.build_index();

        Self::from_hnsw_builder(builder, max_elements, num_threads)
    }

    pub fn from_hnsw_builder(
        builder: HnswBuilder<'_, Elements, Element>,
        max_elements: usize,
        num_threads: usize,
    ) -> Self {
        let mut builder = builder;

        let mut current_layer = builder
            .layers
            .pop()
            .unwrap_or_else(|| FixedWidthSliceVector::new(builder.config.num_neighbors));

        let layer_multiplier = compute_layer_multiplier(max_elements, builder.config.num_layers);
        let num_elements_in_layer = std::cmp::max(
            current_layer.len(),
            std::cmp::min(
                layer_multiplier.powf(builder.layers.len() as f32).ceil() as usize,
                max_elements,
            ),
        );

        current_layer.resize(num_elements_in_layer, UNUSED);

        Self {
            layers: RwLock::new((current_layer.into(), builder.layers)),
            elements: RwLock::new(builder.elements.into_owned()),
            config: builder.config,
            max_elements,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Could not create threadpool"),
            write_lock: RwLock::new(()),
        }
    }

    pub fn save_index_and_elements_to_disk(
        self: &Self,
        index_path: &str,
        elements_path: &str,
        compress: bool,
    ) -> Result<()> {
        let mut elements_file = File::create(elements_path)?;
        let mut index_file = File::create(index_path)?;

        self.write(&mut index_file, &mut elements_file, compress)
    }

    pub fn write(self: &Self, index_file: impl Write + Seek, elements_file: impl Write, compress: bool) -> Result<()> {
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

        let layers = super::Layers::FixWidth({
            if elements.len() > 0 {
                let mut layers: Vec<_> = layers.iter().map(|layer| layer.borrow()).collect();

                layers.push(last_layer.subslice(0, elements.len()));

                layers
            } else {
                vec![]
            }
        });

        io::save_index_to_disk(&layers, index_file, compress)
    }

    pub fn insert(self: &Self, element: QElement) -> Option<usize> {
        self.insert_batch(vec![element]).pop()
    }

    pub fn insert_batch(self: &Self, mut elements_to_insert: Vec<QElement>) -> Vec<usize> {
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
                        std::mem::replace(current_layer, FixedWidthSliceVector::new(1).into()).into();
                    let mut new_layer = current_layer.clone();
                    prev_layers.push(current_layer);

                    let layer_multiplier = compute_layer_multiplier(self.max_elements, self.config.num_layers);
                    let num_elements_in_layer = std::cmp::min(
                        layer_multiplier.powf(prev_layers.len() as f32).ceil() as usize,
                        self.max_elements,
                    );
                    new_layer.resize(num_elements_in_layer, UNUSED);

                    new_layer.into()
                };
            }

            assert!(current_layer.len() >= elements.len());

            // insert elements that fit in the current layer (the remaining will be inserted in a recursive call)
            let num_to_insert = std::cmp::min(elements_to_insert.len(), current_layer.len() - elements.len());
            let ids: Vec<usize> = (elements.len()..).take(num_to_insert).collect();

            let remaining = elements_to_insert.split_off(num_to_insert);

            for element in elements_to_insert {
                elements.append(element);
            }

            // downgrade to read locks before inserting into graph
            let (ref current_layer, ref layers) = *parking_lot::RwLockWriteGuard::downgrade(layers);
            let elements = parking_lot::RwLockWriteGuard::downgrade(elements);

            let index = Hnsw::new(
                Layers::FixWidth(layers.iter().map(|layer| layer.borrow()).collect()),
                &*elements,
            );

            if self.pool.current_num_threads() > 1 {
                self.pool.install(|| {
                    ids.par_iter().for_each(|id| {
                        HnswBuilder::index_element(&self.config, &*elements, &index, current_layer.as_slice(), *id)
                    })
                });
            } else {
                ids.iter().for_each(|id| {
                    HnswBuilder::index_element(&self.config, &*elements, &index, current_layer.as_slice(), *id)
                })
            }

            // locks go out of scope and are released before potentially inserting the remaining elements
            (ids, remaining)
        };

        if !remaining.is_empty() {
            let ids = self.insert_batch(remaining);
            inserted.extend(ids.into_iter());
        }

        inserted
    }

    pub fn search(self: &Self, element: &Element, num_neighbors: usize, max_search: usize) -> Vec<(usize, f32)> {
        let elements = self.elements.read();
        let (ref current_layer, ref layers) = *self.layers.read();

        let index = Hnsw::new(
            Layers::FixWidth(layers.iter().map(|layer| layer.borrow()).collect()),
            &*elements,
        );

        if let Some((entrypoint, _)) = index.search(&element, 1, 1).first() {
            HnswBuilder::search_for_neighbors_index(
                &*elements,
                current_layer.as_slice(),
                *entrypoint,
                element,
                max_search,
            )
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

    pub fn get_element(self: &Self, idx: usize) -> Element {
        self.elements.read().at(idx)
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
    use crate::query_embeddings::*;
    use crate::types::*;
    use std::io::SeekFrom;
    use tempfile;

    #[test]
    fn insert_in_parallel() {
        let config = Config {
            num_layers: 6,
            num_neighbors: 20,
            max_search: 50,
            reinsert_elements: false,
            show_progress: false,
        };

        let builder = RwHnswBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
            config,
            QueryEmbeddings::new(crate::query_embeddings::example::get_random_word_embeddings(5, 10000)),
            5000,
            1,
        );

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
                    builder.search(&crate::types::example::random_dense_element(5), 5, 50);
                });
            },
        );
    }

    #[test]
    fn insert_batch() {
        let config = Config {
            num_layers: 6,
            num_neighbors: 20,
            max_search: 50,
            reinsert_elements: false,
            show_progress: false,
        };

        let max_elements = 1500;
        let num_threads = 4;

        let builder = RwHnswBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
            config,
            QueryEmbeddings::new(crate::query_embeddings::example::get_random_word_embeddings(5, 10000)),
            max_elements,
            num_threads,
        );

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
            assert_eq!(i, builder.search(&builder.get_element(i), 1, 10)[0].0);
        }
    }

    #[test]
    fn layer_counts() {
        const DIM: usize = 2;
        let num_threads = 1;

        for num_layers in vec![2, 5, 6] {
            for max_elements in vec![13, 66, 199, 719] {
                let config = Config {
                    num_layers,
                    num_neighbors: 10,
                    max_search: 50,
                    reinsert_elements: false,
                    show_progress: false,
                };

                let rw_builder =
                    RwHnswBuilder::<AngularVectors<'static>, AngularVector<'static>, AngularVector<'static>>::new(
                        config.clone(),
                        AngularVectors::new(DIM),
                        max_elements,
                        num_threads,
                    );

                for _ in 0..max_elements {
                    rw_builder
                        .insert(crate::types::example::random_dense_element(DIM))
                        .unwrap();
                }

                let elements = rw_builder.elements.read().clone();

                let mut builder = HnswBuilder::<AngularVectors<'static>, AngularVector<'static>>::with_owned_elements(
                    config.clone(),
                    elements,
                );
                builder.build_index();

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
        let config = Config {
            num_layers: 6,
            num_neighbors: 20,
            max_search: 50,
            reinsert_elements: true,
            show_progress: false,
        };

        let builder = RwHnswBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
            config,
            QueryEmbeddings::new(crate::query_embeddings::example::get_random_word_embeddings(5, 10000)),
            5000,
            1,
        );

        builder.search(&crate::types::example::random_dense_element(5), 5, 50);
    }

    #[test]
    fn search_with_one_element() {
        let config = Config {
            num_layers: 6,
            num_neighbors: 20,
            max_search: 50,
            reinsert_elements: true,
            show_progress: false,
        };

        let builder = RwHnswBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
            config,
            QueryEmbeddings::new(crate::query_embeddings::example::get_random_word_embeddings(5, 10000)),
            5000,
            1,
        );

        builder.insert(vec![1, 2, 3]);
        builder.search(&crate::types::example::random_dense_element(5), 5, 50);
    }

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

        let builder = RwHnswBuilder::<QueryEmbeddings<'static>, AngularVector<'static>, Vec<usize>>::new(
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
                HnswBuilder::<crate::query_embeddings::QueryEmbeddings, AngularVector<'static>>::read_index_with_borrowed_elements(config.clone(), &mut index_file, &elements).unwrap();

            assert_eq!(builder.len(), read_builder.indexed_elements());

            let read_builder = RwHnswBuilder::from_hnsw_builder(read_builder, max_elements, num_threads);

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
}
