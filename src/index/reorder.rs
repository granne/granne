use super::*;
use pbr::ProgressBar;

pub(super) fn reorder_layers(
    layers: &Layers,
    mapping: &[usize],
    show_progress: bool,
) -> Layers<'static> {
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
        Some(ProgressBar::new(layer.len() as u64))
    } else {
        None
    };

    let mut new_layer = MultiSetVector::new();

    let chunk_size = std::cmp::max(10_000, layer.len() / 400);
    let chunks: Vec<_> = mapping[..layer.len()]
        .par_chunks(chunk_size)
        .map(|c| {
            let mut layer_chunk = Vec::new();

            for &id in c {
                let neighbors: Vec<NeighborId> = layer
                    .get_neighbors(id)
                    .into_iter()
                    .map(|n| NeighborId::try_from(reverse_mapping[n]).unwrap())
                    .collect();

                layer_chunk.push(neighbors);
            }

            layer_chunk
        })
        .collect();

    for chunk in chunks {
        for c in chunk {
            new_layer.push(&c);
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
    use crate::elements::{embeddings, Dist};
    use crate::test_helper;
    /*
        #[test]
        fn reorder_index_with_sum_embeddings() {
            let elements = test_helper::random_sum_embeddings(5, 277, 100);

            let mut builder = GranneBuilder::new(
                BuildConfig::default().max_search(5).num_neighbors(5),
                elements.borrow(),
            );

            builder.build();

            let index = builder.get_index();

            let layer_counts: Vec<usize> = (0..index.num_layers())
                .map(|layer| index.layer_len(layer))
                .collect();
            let mapping =
                embeddings::reorder::find_reordering_based_on_embeddings(&elements, &layer_counts);

            let reordered_elements =
                embeddings::reorder::reorder_sum_embeddings(&elements, &mapping, false);

            let reordered_index = index.reordered_index(&mapping, &reordered_elements, false);

            assert_eq!(index.num_layers(), reordered_index.num_layers());

            for layer in 0..index.num_layers() {
                assert_eq!(index.layer_len(layer), reordered_index.layer_len(layer));
            }

            for &id in &[0usize, 12, 55] {
                assert!(
                    index
                        .get_element(mapping[id])
                        .dist(&reordered_index.get_element(id))
                        .into_inner()
                        < 0.00001f32
                );

                let element = index.get_element(id);

                let res: Vec<_> = index
                    .search(&element, 10, 5)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect();
                let reordered_res: Vec<_> = reordered_index
                    .search(&element, 10, 5)
                    .into_iter()
                    .map(|(i, _)| mapping[i])
                    .collect();

                assert_eq!(res, reordered_res);
            }
        }
    */
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
