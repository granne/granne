#![allow(missing_docs)]

use super::*;
use crate::elements::angular_int;
use crate::io::Writeable;

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

pub fn parse_elements_and_save_to_disk(
    elements_path: &Path,
    words_path: &Path,
    output_path: &Path,
    show_progress: bool,
) -> usize {
    let word_ids = parse_words(words_path);
    let elements = parsing::parse_elements_in_directory_or_file(elements_path, &word_ids, show_progress);

    let file = File::create(&output_path).unwrap();
    let mut file = BufWriter::new(file);

    elements.write(&mut file).expect("Failed to write elements to disk");

    elements.len()
}

fn get_shard_name(output_path: &Path, shard_id: usize) -> PathBuf {
    if output_path.is_dir() {
        output_path.join(format!("elements-{}.bin", shard_id))
    } else {
        output_path.with_file_name(format!(
            "{}-{}.{}",
            output_path.file_stem().map(|x| x.to_str().unwrap()).unwrap(),
            shard_id,
            output_path.extension().map(|x| x.to_str().unwrap()).unwrap_or("bin")
        ))
    }
}

pub fn parse_elements_and_save_shards_to_disk(
    elements_path: &Path,
    words_path: &Path,
    output_path: &Path,
    num_shards: usize,
    show_progress: bool,
) -> usize {
    let word_ids = parse_words(words_path);

    let elements = parsing::parse_elements_in_directory_or_file(elements_path, &word_ids, show_progress);

    let shard_size = (elements.len() + num_shards - 1) / num_shards;

    for shard in 0..num_shards {
        let output_path = get_shard_name(output_path, shard);

        let begin = shard * shard_size;
        let end = std::cmp::min((shard + 1) * shard_size, elements.len());
        if show_progress {
            println!(
                "Writing elements [{}, {}] to {}",
                begin,
                end,
                output_path.to_str().unwrap()
            );
        }

        let file =
            File::create(&output_path).expect(&format!("Could not create file at {}", output_path.to_str().unwrap()));
        let mut file = BufWriter::new(file);

        elements
            .write_range(&mut file, begin, end)
            .expect("Failed to write elements to disk");
    }

    elements.len()
}

// TODO: Make it possible to write i8 and f32 vectors
pub fn compute_embeddings_and_save_to_disk(
    elements_path: &Path,
    word_embeddings_path: &Path,
    output_path: &Path,
    show_progress: bool,
) {
    let word_embeddings = File::open(word_embeddings_path).expect("Could not open word_embeddings file");
    let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };
    let embeddings = Embeddings::from_bytes(&word_embeddings);

    let elements = File::open(&elements_path).expect("Could not open elements file");
    let elements = unsafe { memmap::Mmap::map(&elements).unwrap() };
    let elements = Elements::from_bytes(&elements);

    let elements = SumEmbeddings::from_parts(embeddings, elements);

    let file = File::create(&output_path).expect("Could not create output file");
    let mut file = BufWriter::new(file);

    let mut progress_bar = if show_progress {
        Some(pbr::ProgressBar::new(elements.len() as u64))
    } else {
        None
    };

    // generate vectors in chunks to limit memory usage
    let num_chunks = 100;
    let chunk_size = (elements.len() + num_chunks - 1) / num_chunks;
    for i in 0..num_chunks {
        let chunk = (i * chunk_size..cmp::min((i + 1) * chunk_size, elements.len())).collect::<Vec<_>>();
        let vectors_vec: Vec<angular_int::Vector> =
            chunk.par_iter().map(|&i| elements.get_embedding(i).into()).collect();

        let mut vectors = angular_int::Vectors::new();

        for v in vectors_vec {
            vectors.push(&v);
        }

        if i == 0 {
            vectors.write(&mut file).unwrap();
        } else {
            crate::io::write_as_bytes(vectors.as_slice(), &mut file).unwrap();
        }

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.add(vectors.len() as u64);
        }
    }
}

pub fn parse_elements_in_directory_or_file(
    path: &Path,
    word_ids: &HashMap<String, usize>,
    show_progress: bool,
) -> Elements<'static> {
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

    let query_parts: Vec<Elements> = parts
        .par_iter()
        .map(|part| {
            let query_file = File::open(&part).expect(&format!("Input file: {:?} not found", &part));

            let elements = if part.to_str().unwrap().ends_with(".gz") {
                let query_file = flate2::read::GzDecoder::new(query_file);
                parse_file(query_file, &word_ids)
            } else {
                parse_file(query_file, &word_ids)
            };

            if let Some(ref progress_bar) = progress_bar {
                progress_bar.lock().inc();
            }

            elements
        })
        .collect();

    let mut progress_bar = if let Some(ref progress_bar) = progress_bar {
        progress_bar.lock().finish_println("All parts parsed\n");
        println!("Collecting elements...");

        Some(pbr::ProgressBar::new(parts.len() as u64))
    } else {
        None
    };

    let mut elements = Elements::new();
    for query_part in query_parts {
        elements.extend_from_slice_vector(&query_part);

        if let Some(ref mut progress_bar) = progress_bar {
            progress_bar.inc();
        }
    }

    if let Some(ref mut progress_bar) = progress_bar {
        progress_bar.finish_println("Elements collected.\n");
    }

    elements
}

fn parse_file<T: Read>(query_file: T, word_ids: &HashMap<String, usize>) -> Elements<'static> {
    let query_file = BufReader::new(query_file);

    let mut elements = Elements::new();

    for qs in query_file.lines() {
        let mut query_data = Vec::new();

        let qs = serde_json::from_str::<String>(&qs.unwrap()).unwrap();
        let qs = qs.split(':').last().unwrap();

        for word in qs.split_whitespace() {
            if let Some(&id) = word_ids.get(word) {
                query_data.push(EmbeddingId::from(id));
            }
        }

        elements.push(&query_data);
    }

    elements
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn get_shard_name_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        assert_eq!(path.join("elements-4.bin"), get_shard_name(&path, 4));
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
