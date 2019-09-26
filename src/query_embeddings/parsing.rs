use crate::query_embeddings::*;
use crate::types::{AngularVector, AngularVectorT};

use flate2;
use memmap;
use parking_lot::Mutex;
use pbr;
use rayon::prelude::*;
use serde_json;
use std::cmp;
use std::collections::HashMap;
use std::fs::{read_dir, File};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};

pub fn parse_words(words_path: &Path) -> HashMap<String, usize> {
    let word_file = File::open(&words_path).unwrap();
    let word_file = BufReader::new(word_file);

    word_file
        .lines()
        .enumerate()
        .map(|(i, w)| {
            let w = w.unwrap();
            (serde_json::from_str::<String>(&w).unwrap(), i)
        })
        .collect()
}

pub fn parse_queries_and_save_to_disk(
    queries_path: &Path,
    words_path: &Path,
    output_path: &Path,
    show_progress: bool,
) -> usize {
    let word_ids = parse_words(words_path);
    let queries = parsing::parse_queries_in_directory_or_file(queries_path, &word_ids, show_progress);

    let file = File::create(&output_path).unwrap();
    let mut file = BufWriter::new(file);

    queries.write(&mut file).expect("Failed to write queries to disk");

    queries.len()
}

fn get_shard_name(output_path: &Path, shard_id: usize) -> PathBuf {
    if output_path.is_dir() {
        output_path.join(format!("queries-{}.bin", shard_id))
    } else {
        output_path.with_file_name(format!(
            "{}-{}.{}",
            output_path.file_stem().map(|x| x.to_str().unwrap()).unwrap(),
            shard_id,
            output_path.extension().map(|x| x.to_str().unwrap()).unwrap_or("bin")
        ))
    }
}

pub fn parse_queries_and_save_shards_to_disk(
    queries_path: &Path,
    words_path: &Path,
    output_path: &Path,
    num_shards: usize,
    show_progress: bool,
) -> usize {
    let word_ids = parse_words(words_path);

    let queries = parsing::parse_queries_in_directory_or_file(queries_path, &word_ids, show_progress);

    let shard_size = (queries.len() + num_shards - 1) / num_shards;

    for shard in 0..num_shards {
        let output_path = get_shard_name(output_path, shard);

        let begin = shard * shard_size;
        let end = std::cmp::min((shard + 1) * shard_size, queries.len());
        if show_progress {
            println!(
                "Writing queries [{}, {}] to {}",
                begin,
                end,
                output_path.to_str().unwrap()
            );
        }

        let file =
            File::create(&output_path).expect(&format!("Could not create file at {}", output_path.to_str().unwrap()));
        let mut file = BufWriter::new(file);

        queries
            .queries
            .write_range(&mut file, begin, end)
            .expect("Failed to write queries to disk");
    }

    queries.len()
}

pub fn compute_query_vectors_and_save_to_disk<DTYPE: 'static + Copy + Sync + Send>(
    dimension: usize,
    queries_path: &Path,
    word_embeddings_path: &Path,
    output_path: &Path,
    show_progress: bool,
) where
    AngularVectorT<'static, DTYPE>: From<AngularVector<'static>>,
{
    let word_embeddings = File::open(word_embeddings_path).expect("Could not open word_embeddings file");
    let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

    let queries = File::open(&queries_path).expect("Could not open queries file");
    let queries = unsafe { memmap::Mmap::map(&queries).unwrap() };

    let queries = QueryEmbeddings::load(dimension, &word_embeddings, &queries);

    let file = File::create(&output_path).expect("Could not create output file");
    let mut file = BufWriter::new(file);

    let mut progress_bar = if show_progress {
        Some(pbr::ProgressBar::new(queries.len() as u64))
    } else {
        None
    };

    // generate vectors in chunks to limit memory usage
    let num_chunks = 100;
    let chunk_size = (queries.len() + num_chunks - 1) / num_chunks;
    for i in 0..num_chunks {
        let chunk = (i * chunk_size..cmp::min((i + 1) * chunk_size, queries.len())).collect::<Vec<_>>();
        let query_vectors: Vec<AngularVectorT<'static, DTYPE>> =
            chunk.par_iter().map(|&i| queries.at(i).into()).collect();

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.add(query_vectors.len() as u64);
        }
        for q in query_vectors {
            file_io::write(q.as_slice(), &mut file).unwrap();
        }
    }
}

pub fn parse_queries_in_directory_or_file(
    path: &Path,
    word_ids: &HashMap<String, usize>,
    show_progress: bool,
) -> QueryVec<'static> {
    let parts = if path.is_dir() {
        let mut parts: Vec<_> = read_dir(path).unwrap().map(|p| p.unwrap().path()).collect();
        parts.sort();
        parts
    } else {
        vec![path.to_path_buf()]
    };

    let progress_bar = if show_progress {
        println!("Parsing {} part(s)...", parts.len());
        Some(Mutex::new(pbr::ProgressBar::new(parts.len() as u64)))
    } else {
        None
    };

    let query_parts: Vec<QueryVec> = parts
        .par_iter()
        .map(|part| {
            let query_file = File::open(&part).expect(&format!("Input file: {:?} not found", &part));

            let queries = if part.to_str().unwrap().ends_with(".gz") {
                let query_file = flate2::read::GzDecoder::new(query_file).expect("Not a valid gzip file.");
                parse_file(query_file, &word_ids)
            } else {
                parse_file(query_file, &word_ids)
            };

            if let Some(ref progress_bar) = progress_bar {
                progress_bar.lock().inc();
            }

            queries
        })
        .collect();

    let mut progress_bar = if let Some(ref progress_bar) = progress_bar {
        progress_bar.lock().finish_println("All parts parsed\n");
        println!("Collecting queries...");

        Some(pbr::ProgressBar::new(parts.len() as u64))
    } else {
        None
    };

    let mut queries = QueryVec::new();
    for query_part in query_parts {
        queries.extend_from_queryvec(&query_part);

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.inc();
        }
    }

    if let Some(ref mut progress_bar) = progress_bar {
        progress_bar.finish_println("Queries collected.\n");
    }

    queries
}

fn parse_file<T: Read>(query_file: T, word_ids: &HashMap<String, usize>) -> QueryVec<'static> {
    let query_file = BufReader::new(query_file);

    let mut queries = QueryVec::new();

    for qs in query_file.lines() {
        let mut query_data = Vec::new();

        let qs = serde_json::from_str::<String>(&qs.unwrap()).unwrap();
        let qs = qs.split(':').last().unwrap();

        for word in qs.split_whitespace() {
            if let Some(&id) = word_ids.get(word) {
                query_data.push(id);
            }
        }

        queries.push(&query_data);
    }

    queries
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn get_shard_name_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        assert_eq!(path.join("queries-4.bin"), get_shard_name(&path, 4));
    }

    #[test]
    fn get_shard_name_with_filename() {
        assert_eq!(
            Path::new("/output/directory/hello-4.world"),
            get_shard_name(&Path::new("/output/directory/hello.world"), 4)
        );
    }

    #[test]
    fn get_shard_name_with_filestem() {
        assert_eq!(
            Path::new("/output/directory/hello-9.bin"),
            get_shard_name(&Path::new("/output/directory/hello"), 9)
        );
    }
}
