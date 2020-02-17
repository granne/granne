use super::*;

impl<'a, Elements: ElementContainer + Permutable + Sync> Granne<'a, Elements> {
    /** Reorders the elements in this index. Tries to place similar elements closer together in the
    graph. Works for any type of elements. Note, however, that results are usually not as good
    as when using [`embeddings::reorder::compute_keys_for_reordering`](embeddings/fn.compute_keys_for_reordering.html).

    Returns the permutation used for the  reordering. `permutation[i] == j`, means that the
    element with idx `j`, has been moved to idx `i`.

    Reordering of the index is not required but can be useful in a couple of situation:
     * Since the neighbors of each node in the index is stored using a variable int encoding, reordering might make size of index smaller.
     * Improve data locality for serving the index from disk. See
    [Indexing Billions of Text Vectors](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html)
    for a more thorough explanation of this use case.

    ## Example usage

    ```
    # use granne::{Granne, GranneBuilder, Index, Builder, BuildConfig, angular};
    # use tempfile;
    # const DIM: usize = 5;
    # fn main() -> std::io::Result<()> {
    # let elements: angular::Vectors = granne::test_helper::random_vectors(DIM, 1000);
    # let random_vector: angular::Vector = granne::test_helper::random_vector(DIM);
    # let num_results = 10;
    #
    # let mut builder = GranneBuilder::new(BuildConfig::default(), elements);
    # builder.build();
    # let mut index_file = tempfile::tempfile()?;
    # builder.write_index(&mut index_file)?;
    # let mut elements_file = tempfile::tempfile()?;
    # builder.write_elements(&mut elements_file)?;
    # let max_search = 10;
    # let num_neighbors = 10;
    use granne::{angular, Granne};

    // loading index and vectors (original)
    let elements = unsafe { angular::Vectors::from_file(&elements_file)? };
    let index = unsafe { Granne::from_file(&index_file, elements)? };

    // loading index and vectors (for reordering)
    let elements = unsafe { angular::Vectors::from_file(&elements_file)? };
    let mut reordered_index = unsafe { Granne::from_file(&index_file, elements)? };
    let order = reordered_index.reorder(false);

    // verify that results are the same
    let element = index.get_element(123);
    let res = index.search(&element, max_search, num_neighbors);
    let reordered_res = reordered_index.search(&element, max_search, num_neighbors);
    for (r, rr) in res.iter().zip(&reordered_res) {
        assert_eq!(r.0, order[rr.0]);
    }

    # Ok(())
    # }
    ```
     */
    pub fn reorder(self: &mut Self, show_progress: bool) -> Vec<usize> {
        let order = self.compute_order(show_progress);

        let start_time = time::Instant::now();

        if show_progress {
            println!("Order computed!");
            println!("Reordering index...");
        }

        self.layers = FileOrMemoryLayers::Memory(reorder_layers(&self.layers.load(), &order, show_progress));

        if show_progress {
            println!("Reordering elements...");
        }

        self.elements.permute(&order);

        if show_progress {
            println!("Reordered index and elements in {} s", start_time.elapsed().as_secs());
        }

        order
    }

    /// Essentially a layer-preserving sort, i.e., it reorders the elements in this index based on
    /// keys while respecting layers. This means that the new positions of elements in layer `i`
    /// will be in the range `[0, layer_len(i)]`.
    ///
    /// Returns the permutation used for the reordering. `permutation[i] == j`, means that the
    /// element with idx `j`, has been moved to idx `i`.
    pub fn reorder_by_keys(self: &mut Self, keys: &[impl Ord + Sync + Send], show_progress: bool) -> Vec<usize> {
        let start_time = time::Instant::now();

        assert_eq!(self.len(), keys.len());

        let layer_lens: Vec<usize> = (0..self.num_layers()).map(|i| self.layer_len(i)).collect();

        let mut order = Vec::new();
        for layer in 0..layer_lens.len() {
            let begin = if layer > 0 { layer_lens[layer - 1] } else { 0 };
            let end = layer_lens[layer];

            let mut layer_order: Vec<_> = (begin..end).into_par_iter().map(|l| (&keys[l], l)).collect();
            layer_order.par_sort_unstable();
            order.par_extend(layer_order.into_par_iter().map(|(_key, l)| l));
        }

        if show_progress {
            println!("Order computed!");
            println!("Reordering index...");
        }

        self.layers = FileOrMemoryLayers::Memory(reorder_layers(&self.layers.load(), &order, show_progress));

        if show_progress {
            println!("Reordering elements...");
        }

        self.elements.permute(&order);

        if show_progress {
            println!("Total time: {} s", start_time.elapsed().as_secs());
        }

        order
    }

    fn compute_order(self: &mut Self, show_progress: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.layer_len(0)).collect();
        let mut order_inv = vec![0; self.layer_len(self.num_layers() - 2)];

        let start_time = time::Instant::now();
        let progress_bar = if show_progress {
            println!("Computing order...");

            Some(parking_lot::Mutex::new(pbr::ProgressBar::new(self.len() as u64)))
        } else {
            None
        };

        let step_size = cmp::max(2500, self.len() / 1000);
        for layer in 1..self.num_layers() {
            let mut eps: Vec<_> = (self.layer_len(layer - 1)..self.layer_len(layer))
                .into_par_iter()
                .map(|idx| {
                    if idx % step_size == 0 {
                        progress_bar.as_ref().map(|pb| pb.lock().add(step_size as u64));
                    }

                    let mut eps =
                        find_entrypoint_trail(&self.layers.load(), &self.elements, layer, &self.get_element(idx));
                    eps.iter_mut().for_each(|i| *i = order_inv[*i as usize] as NeighborId);

                    (eps, idx)
                })
                .collect();

            eps.par_sort_unstable();
            order.par_extend(eps.into_par_iter().map(|(_key, i)| i));

            if layer < self.num_layers() - 1 {
                for i in self.layer_len(layer - 1)..self.layer_len(layer) {
                    order_inv[order[i]] = i;
                }
            }
        }

        progress_bar.map(|pb| pb.lock().set(self.len() as u64));

        if show_progress {
            println!("Order computed in {} s", start_time.elapsed().as_secs());
        }

        order
    }
}

const NUM_LAYERS: usize = 8;

/// Returns an array containing the ids of the "closest" element in each layer.
fn find_entrypoint_trail<Elements: ElementContainer>(
    layers: &Layers,
    elements: &Elements,
    max_layer: usize,
    element: &Elements::Element,
) -> [NeighborId; NUM_LAYERS] {
    fn _find_entrypoint_trail<Layer: Graph, Elements: ElementContainer>(
        layers: &[Layer],
        elements: &Elements,
        max_layer: usize,
        element: &Elements::Element,
    ) -> [NeighborId; NUM_LAYERS] {
        let mut eps: [NeighborId; NUM_LAYERS] = [0; NUM_LAYERS];
        for (i, layer) in layers.iter().enumerate().take(cmp::min(NUM_LAYERS, max_layer)) {
            let ep = if i == 0 { 0 } else { eps[i] };
            let max_search = 1;
            let res = search_for_neighbors(layer, ep.try_into().unwrap(), elements, element, max_search);
            use std::convert::TryInto;
            eps[i] = res[0].0.try_into().unwrap();
        }
        eps
    }

    match layers {
        Layers::FixWidth(layers) => _find_entrypoint_trail(layers, elements, max_layer, element),
        Layers::Compressed(layers) => _find_entrypoint_trail(layers, elements, max_layer, element),
    }
}

fn reorder_layers(layers: &Layers, mapping: &[usize], show_progress: bool) -> Layers<'static> {
    let reverse_mapping = get_reverse_mapping(mapping);
    match layers {
        Layers::FixWidth(ref layers) => Layers::Compressed(
            layers
                .iter()
                .map(|layer| reorder_layer(layer, mapping, &reverse_mapping, show_progress))
                .collect(),
        ),
        Layers::Compressed(ref layers) => Layers::Compressed(
            layers
                .iter()
                .map(|layer| reorder_layer(layer, mapping, &reverse_mapping, show_progress))
                .collect(),
        ),
    }
}

fn reorder_layer<Layer: Graph + Sync + Send>(
    layer: &Layer,
    mapping: &[usize],
    reverse_mapping: &[usize],
    show_progress: bool,
) -> MultiSetVector<'static> {
    let mut progress_bar = if show_progress {
        Some(pbr::ProgressBar::new(layer.len() as u64))
    } else {
        None
    };

    let mut new_layer = MultiSetVector::new();

    let chunk_size = std::cmp::max(10_000, layer.len() / 1000);
    let chunks: Vec<_> = mapping[..layer.len()]
        .par_chunks(chunk_size)
        .map(|c| {
            let mut offset_chunk = vec![0];
            let mut layer_chunk = Vec::new();

            for &id in c {
                layer_chunk.extend(
                    layer
                        .get_neighbors(id)
                        .into_iter()
                        .map(|n| NeighborId::try_from(reverse_mapping[n]).unwrap()),
                );

                offset_chunk.push(layer_chunk.len());
            }

            (offset_chunk, layer_chunk)
        })
        .collect();

    for (offset, chunk) in chunks {
        for i in 1..offset.len() {
            new_layer.push(&chunk[offset[i - 1]..offset[i]]);
        }

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.set(new_layer.len() as u64);
        }
    }

    if let Some(ref mut progress_bar) = progress_bar {
        progress_bar.finish_println("");
    }

    new_layer
}

fn get_reverse_mapping(mapping: &[usize]) -> Vec<usize> {
    let mut rev_mapping = Vec::new();
    rev_mapping.resize(mapping.len(), 0);

    for (i, &j) in mapping.iter().enumerate() {
        rev_mapping[j] = i;
    }

    rev_mapping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::angular;
    use crate::test_helper;

    #[test]
    fn reorder_index() {
        let elements: angular::Vectors = (0..5000)
            .map(|_| test_helper::random_vector::<angular::Vector>(5))
            .collect();

        let mut builder = GranneBuilder::new(
            BuildConfig::default().max_search(5).layer_multiplier(5.0),
            elements.clone(),
        );

        builder.build();

        let mut reordered_index = Granne::from_parts(builder.layers.clone(), elements);
        let permutation = reordered_index.reorder(false);

        let index = builder.get_index();

        for &idx in &[0, 10, 123, 99, 499] {
            let element = index.get_element(idx);
            let exp = index.search(&element, 10, 10);
            let res = reordered_index.search(&element, 10, 10);
            for i in 0..10 {
                assert_eq!(exp[i].0, permutation[res[i].0]);
            }
        }
    }

    #[test]
    fn test_reverse_mapping() {
        let mapping: Vec<_> = (0..105).rev().collect();
        let rev_mapping = get_reverse_mapping(&mapping);

        assert_eq!(mapping.len(), rev_mapping.len());

        for i in 0..mapping.len() {
            assert_eq!(i, rev_mapping[mapping[i]]);
        }
    }
}
