use super::{PyGranne, PyGranneBuilder, SaveIndex};

use madvise::{AccessPattern, AdviseMemory};
use memmap;
use std::collections::HashMap;
use std::io::BufRead;

pub mod builder;
pub mod index;

fn open_random_access_mmap(path: &str) -> memmap::Mmap {
    let file = std::fs::File::open(path).expect(&format!("Could not open file at \"{}\".", path));
    let file = unsafe { memmap::Mmap::map(&file).expect("Mmap failed!") };
    file.advise_memory_access(AccessPattern::Random)
        .expect("Error with madvise");
    file
}

struct WordDict {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl WordDict {
    pub fn new(path: &str) -> Self {
        let word_file =
            std::fs::File::open(&path).expect(&format!("Could not open file at \"{}\"!", path));
        let word_file = std::io::BufReader::new(word_file);
        let words: Vec<String> = word_file
            .lines()
            .map(|w| {
                let w = w.unwrap();
                serde_json::from_str::<String>(&w).unwrap()
            })
            .collect();

        let word_to_id = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i))
            .collect();

        Self {
            word_to_id: word_to_id,
            id_to_word: words,
        }
    }

    pub fn len(self: &Self) -> usize {
        self.id_to_word.len()
    }

    pub fn get_words(self: &Self, ids: &[usize]) -> String {
        if ids.is_empty() {
            return String::new();
        }

        let mut query = self.id_to_word[ids[0]].clone();

        for word in ids[1..].iter().map(|&id| self.id_to_word[id].as_str()) {
            query.push(' ');
            query.push_str(word);
        }

        query
    }

    pub fn get_word_ids(self: &Self, query: &str) -> Vec<usize> {
        query
            .split_whitespace()
            .filter_map(|w| self.word_to_id.get(w).cloned())
            .collect()
    }
}
