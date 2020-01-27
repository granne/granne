use super::*;
use ordered_float::NotNan;
use rayon::prelude::*;

/// Finds a reordering of queries by sorting elements based on their embedding id with largest norm
/// while respecting the Hnsw layer structure
pub fn find_reordering_based_on_embeddings<'a>(
    sum_embeddings: &SumEmbeddings<'a>,
    layer_counts: &[usize],
) -> Vec<usize> {
    let mut mapping = vec![0];
    for &layer_count in layer_counts {
        mapping = find_reordering_in_last_layer(sum_embeddings, layer_count, mapping);
    }

    mapping
}

fn find_reordering_in_last_layer<'a>(
    sum_embeddings: &SumEmbeddings<'a>,
    layer_count: usize,
    started: Vec<usize>,
) -> Vec<usize> {
    // get norms of word embeddings
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let w_norms: Vec<NotNan<f32>> = (0..sum_embeddings.embeddings.len())
        .into_par_iter()
        .map(|i| norm(&sum_embeddings.create_embedding(&[i])).into())
        .collect();

    const NUM_WORDS: usize = 8;

    // sort words in each query based on "term relevance"
    let get_key = |q| {
        let mut embedding_ids = sum_embeddings.get_terms(q);
        embedding_ids.sort_by_key(|&w| w_norms[w]);
        embedding_ids.reverse();

        let mut key = [0; NUM_WORDS];
        let len = std::cmp::min(embedding_ids.len(), NUM_WORDS);

        key[..len].copy_from_slice(&embedding_ids[..len]);
        key
    };

    let mut mapping: Vec<([usize; NUM_WORDS], usize)> = (started.len()..layer_count)
        .into_par_iter()
        .map(|i| (get_key(i), i))
        .collect();

    // sort queries based on most important words
    mapping.par_sort_unstable();

    let mut started = started;
    started.extend(mapping.into_iter().map(|(_, i)| i));
    started
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::Permutable;
    use crate::test_helper;

    #[test]
    fn find_reordering_respects_layer_counts() {
        let queries = test_helper::random_sum_embeddings(25, 225, 200);
        let layer_counts = vec![1, 5, 15, 25, queries.len()];
        let mut mapping = find_reordering_based_on_embeddings(&queries, &layer_counts);

        assert_eq!(queries.len(), mapping.len());

        for layer_count in layer_counts {
            mapping[..layer_count].sort();
            for i in 0..layer_count {
                assert_eq!(i, mapping[i]);
            }
        }
    }

    #[test]
    fn reorder_sum_embeddings_reverse() {
        let queries = test_helper::random_sum_embeddings(25, 225, 200);
        let mapping: Vec<usize> = (0..queries.len()).rev().collect();

        let mut rev_queries = queries.clone();
        rev_queries.permute(&mapping);

        assert_eq!(queries.len(), rev_queries.len());

        for i in 0..queries.len() {
            assert_eq!(
                queries.get_terms(i),
                rev_queries.get_terms(queries.len() - i - 1)
            );
        }
    }
}
