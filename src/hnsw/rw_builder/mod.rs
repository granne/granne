use super::*;

use parking_lot::{self, RwLock};
use slice_vector::FixedWidthSliceVector;

mod rwlock_slice_vector;

use rwlock_slice_vector::RwLockSliceVector;

pub struct RwHnswBuilder<Elements, Element, QElement>
where
    Elements: At<Output = Element> + Appendable<Element = QElement> + Sync + Send + ToOwned,
    Element: ComparableTo<Element> + Sync + Send,
{
    layers: RwLock<(
        RwLockSliceVector<NeighborId>,
        Vec<FixedWidthSliceVector<'static, NeighborId>>,
    )>,
    elements: RwLock<Elements>,
    config: Config,
    max_elements: usize,
    pool: rayon::ThreadPool,
}

impl<Elements, Element, QElement> RwHnswBuilder<Elements, Element, QElement>
where
    Elements: At<Output = Element> + Appendable<Element = QElement> + Sync + Send + Clone,
    Element: ComparableTo<Element> + Sync + Send,
{
    pub fn new(config: Config, elements: Elements, max_elements: usize, num_threads: usize) -> Self {
        let mut builder = HnswBuilder::with_owned_elements(config, elements);

        builder.build_index();

        Self::from_hnsw_builder(builder, max_elements, num_threads)
    }

    pub fn from_hnsw_builder<'b>(
        builder: HnswBuilder<'b, Elements, Element>,
        max_elements: usize,
        num_threads: usize,
    ) -> Self {
        let mut builder = builder;

        let mut current_layer = builder
            .layers
            .pop()
            .unwrap_or(FixedWidthSliceVector::new(builder.config.num_neighbors));

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
        }
    }

    pub fn insert(self: &Self, element: QElement) -> Option<usize> {
        self.insert_batch(vec![element]).pop()
    }

    pub fn insert_batch(self: &Self, mut elements_to_insert: Vec<QElement>) -> Vec<usize> {
        if self.elements.read().len() >= self.max_elements {
            return vec![];
        }

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

            // locks go out of scope and is released before potentially inserting the remaining elements
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

    #[test]
    fn insert_in_parallel() {
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
}
