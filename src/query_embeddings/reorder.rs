use super::*;
use ordered_float::NotNaN;
use pbr::ProgressBar;
use rayon::prelude::*;

pub fn reorder_query_embeddings<'a>(
    query_embeddings: &QueryEmbeddings<'a>,
    mapping: &[usize],
    show_progress: bool,
) -> QueryEmbeddings<'static> {
    // this deep copy of the word embeddings is necessary as long as the lifetimes of the members of
    // QueryEmbeddings are the same
    let mut reordered = QueryEmbeddings::new(query_embeddings.word_embeddings.clone().into_owned());

    let mut progress_bar = if show_progress {
        Some(ProgressBar::new(query_embeddings.len() as u64))
    } else {
        None
    };

    let chunk_size = std::cmp::max(10_000, query_embeddings.len() / 400);
    let chunks: Vec<_> = mapping
        .par_chunks(chunk_size)
        .map(|c| {
            let mut qe = QueryEmbeddings::new(WordEmbeddings::new());
            for &id in c {
                qe.append(query_embeddings.get_words(id));
            }
            qe
        })
        .collect();

    for chunk in chunks {
        reordered.queries.extend_from_queryvec(&chunk.queries);

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.set(reordered.len() as u64);
        }
    }

    if let Some(ref mut progress_bar) = progress_bar {
        progress_bar.finish_println("");
    }

    reordered
}

/// Finds a reordering of queries by sorting queries based on their word id with highest term
/// relevance while respecting the Hnsw layer structure
pub fn find_reordering_based_on_queries<'a>(
    query_embeddings: &QueryEmbeddings<'a>,
    layer_counts: &[usize],
) -> Vec<usize> {
    let mut mapping = vec![0];
    for &layer_count in layer_counts {
        mapping = find_reordering_in_last_layer(query_embeddings, layer_count, mapping);
    }

    mapping
}

fn find_reordering_in_last_layer<'a>(
    query_embeddings: &QueryEmbeddings<'a>,
    layer_count: usize,
    started: Vec<usize>,
) -> Vec<usize> {
    // get norms of word embeddings
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let w_norms: Vec<NotNaN<f32>> = (0..query_embeddings.word_embeddings.len())
        .into_par_iter()
        .map(|i| norm(&query_embeddings.word_embeddings.get_raw_embedding(&[i])).into())
        .collect();

    const NUM_WORDS: usize = 8;

    // sort words in each query based on "term relevance"
    let get_key = |q| {
        let mut word_ids = query_embeddings.get_words(q);
        word_ids.sort_by_key(|&w| w_norms[w]);
        word_ids.reverse();

        let mut key = [0; NUM_WORDS];
        let len = std::cmp::min(word_ids.len(), NUM_WORDS);

        key[..len].copy_from_slice(&word_ids[..len]);
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
    use super::example::*;
    use super::*;

    #[test]
    fn find_reordering_respects_layer_counts() {
        let queries = get_sample_query_embeddings();
        let layer_counts = vec![1, 5, 15, 25, queries.len()];
        let mut mapping = find_reordering_based_on_queries(&queries, &layer_counts);

        assert_eq!(queries.len(), mapping.len());

        for layer_count in layer_counts {
            mapping[..layer_count].sort();
            for i in 0..layer_count {
                assert_eq!(i, mapping[i]);
            }
        }
    }

    #[test]
    fn reorder_query_embeddings_reverse() {
        let queries = get_sample_query_embeddings();
        let mapping: Vec<usize> = (0..queries.len()).rev().collect();

        let rev_queries = reorder_query_embeddings(&queries, &mapping, false);

        assert_eq!(queries.len(), rev_queries.len());

        for i in 0..queries.len() {
            assert_eq!(queries.get_words(i), rev_queries.get_words(queries.len() - i - 1));
        }
    }
}
