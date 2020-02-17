/*! Reordering based on query embeddings.

To be used together with [`Granne::reorder_by_keys`](../../struct.Granne.html#method.reorder_by_keys).

# Examples
```
# use granne::test_helper;
# use granne::{BuildConfig, GranneBuilder, Granne, Builder, embeddings};
#
# let elements = test_helper::random_sum_embeddings(5, 50, 100);
#
# let mut builder = GranneBuilder::new(
#    BuildConfig::default().max_search(5).layer_multiplier(5.0),
#    elements
# );
#
# builder.build();
# let mut index: Granne<embeddings::SumEmbeddings> = builder.get_index().to_owned();
# /*
let mut index: Granne<embeddings::SumEmbeddings> = /* omitted */
# */

let keys = embeddings::compute_keys_for_reordering(index.get_elements());
let order = index.reorder_by_keys(&keys, false);
 */

use super::*;
use ordered_float::NotNan;
use rayon::prelude::*;

/// Computes keys for the elements based on their embedding id with largest norm.
pub fn compute_keys_for_reordering(embeddings: &SumEmbeddings<'_>) -> Vec<impl Ord + Copy + Sync + Send> {
    // get norms of word embeddings
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let w_norms: Vec<NotNan<f32>> = (0..embeddings.embeddings.len())
        .into_par_iter()
        .map(|i| norm(&embeddings.create_embedding(&[i])).into())
        .collect();

    const NUM_WORDS: usize = 8;

    // sort words in each query based on "term relevance"
    let get_key = |q| {
        let mut embedding_ids = embeddings.get_terms(q);
        embedding_ids.sort_by_key(|&w| w_norms[w]);
        embedding_ids.reverse();

        let mut key = [0; NUM_WORDS];
        let len = std::cmp::min(embedding_ids.len(), NUM_WORDS);

        key[..len].copy_from_slice(&embedding_ids[..len]);
        key
    };

    (0..embeddings.len()).into_par_iter().map(|i| get_key(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;
    use crate::Permutable;
    use crate::{BuildConfig, Builder, Granne, GranneBuilder};

    #[test]
    fn reorder_sum_embeddings_reverse() {
        let queries = test_helper::random_sum_embeddings(25, 225, 200);
        let mapping: Vec<usize> = (0..queries.len()).rev().collect();

        let mut rev_queries = queries.clone();
        rev_queries.permute(&mapping);

        assert_eq!(queries.len(), rev_queries.len());

        for i in 0..queries.len() {
            assert_eq!(queries.get_terms(i), rev_queries.get_terms(queries.len() - i - 1));
        }
    }

    #[test]
    fn reorder_sum_embeddings() {
        let elements = test_helper::random_sum_embeddings(5, 277, 500);

        let mut builder = GranneBuilder::new(BuildConfig::default().max_search(5).layer_multiplier(5.0), elements);

        builder.build();

        let index = builder.get_index();

        let keys = compute_keys_for_reordering(index.get_elements());
        let mut reordered_index: Granne<'static, SumEmbeddings<'static>> = index.to_owned();
        let permutation = reordered_index.reorder_by_keys(&keys, false);

        for &idx in &[0, 10, 123, 99, 499] {
            let element = index.get_element(idx);
            let exp = index.search(&element, 10, 10);
            let res = reordered_index.search(&element, 10, 10);
            for i in 0..10 {
                assert_eq!(exp[i].0, permutation[res[i].0]);
            }
        }
    }
}
