use super::*;
use pbr::ProgressBar;

pub fn get_reverse_mapping(mapping: &[usize]) -> Vec<usize> {
    let mut rev_mapping = Vec::new();
    rev_mapping.resize(mapping.len(), 0);

    for (i, &j) in mapping.iter().enumerate() {
        rev_mapping[j] = i;
    }

    rev_mapping
}

pub fn reorder_index<'a, Elements, Element>(
    index: &Hnsw<'a, Elements, Element>,
    mapping: &[usize],
    show_progress: bool,
) -> Layers<'static>
where
    Elements: At<Output = Element> + Sync + Send + ToOwned + ?Sized,
    Element: ComparableTo<Element> + Sync + Send,
{
    reorder_layers(&index.layers, mapping, show_progress)
}

fn reorder_layers(layers: &Layers, mapping: &[usize], show_progress: bool) -> Layers<'static> {
    let reverse_mapping = get_reverse_mapping(mapping);
    match layers {
        Layers::FixWidth(ref layers) => Layers::VarWidth(
            layers
                .iter()
                .map(|layer| reorder_layer(layer, mapping, &reverse_mapping, show_progress))
                .collect(),
        ),
        Layers::VarWidth(ref layers) => Layers::VarWidth(
            layers
                .iter()
                .map(|layer| reorder_layer(layer, mapping, &reverse_mapping, show_progress))
                .collect(),
        ),
    }
}

fn reorder_layer<Layer: At<Output = Vec<usize>> + Sync + Send>(
    layer: &Layer,
    mapping: &[usize],
    reverse_mapping: &[usize],
    show_progress: bool,
) -> VariableWidthSliceVector<'static, NeighborId, NeighborId> {
    let mut progress_bar = if show_progress {
        Some(ProgressBar::new(layer.len() as u64))
    } else {
        None
    };

    let mut new_layer = VariableWidthSliceVector::new();

    let chunk_size = std::cmp::max(10_000, layer.len() / 400);
    let chunks: Vec<_> = mapping[..layer.len()]
        .par_chunks(chunk_size)
        .map(|c| {
            let mut layer_chunk = new_layer.clone();

            for &id in c {
                let neighbors: Vec<NeighborId> = layer.at(id).into_iter().map(|n| reverse_mapping[n].into()).collect();

                layer_chunk.push(&neighbors);
            }

            layer_chunk
        })
        .collect();

    for chunk in chunks {
        new_layer.extend_from_slice_vector(&chunk);

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.set(new_layer.len() as u64);
        }
    }

    if let Some(ref mut progress_bar) = progress_bar {
        progress_bar.finish_println("");
    }

    new_layer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_embeddings;
    use crate::types::AngularVector;

    #[test]
    fn reorder_index_with_queries() {
        let queries = query_embeddings::example::get_sample_query_embeddings();
        let config = Config {
            num_layers: 5,
            num_neighbors: 5,
            max_search: 5,
            reinsert_elements: true,
            show_progress: false,
        };

        let mut builder = HnswBuilder::with_borrowed_elements(config, &queries);

        builder.build_index();

        let index = builder.get_index();

        let layer_counts: Vec<usize> = (0..index.num_layers()).map(|layer| index.layer_len(layer)).collect();
        let mapping = query_embeddings::reorder::find_reordering_based_on_queries(&queries, &layer_counts);

        let reordered_queries = query_embeddings::reorder::reorder_query_embeddings(&queries, &mapping, false);
        let reordered_layers = reorder_index(&index, &mapping, false);

        let reordered_index = Hnsw::new(reordered_layers, &reordered_queries);

        assert_eq!(index.num_layers(), reordered_index.num_layers());

        for layer in 0..index.num_layers() {
            assert_eq!(index.layer_len(layer), reordered_index.layer_len(layer));
        }

        for &id in &[0usize, 12, 55] {
            assert!(index.get_element(mapping[id]).dist(&reordered_index.get_element(id)) < AngularVector::eps());

            let element = index.get_element(id);

            let res: Vec<_> = index.search(&element, 10, 5).into_iter().map(|(i, _)| i).collect();
            let reordered_res: Vec<_> = reordered_index
                .search(&element, 10, 5)
                .into_iter()
                .map(|(i, _)| mapping[i])
                .collect();

            assert_eq!(res, reordered_res);
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
