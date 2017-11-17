#![feature(iterator_step_by)]


extern crate arrayvec;
extern crate ordered_float;
extern crate time;
extern crate rand;
extern crate rayon;
extern crate fnv;
extern crate memmap;

use types::*;
use std::collections::BinaryHeap;
pub use ordered_float::NotNaN;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::prelude::*;
use memmap::Mmap;

mod types;
mod file_io;
mod hnsw;

const MAX_NEIGHBORS: usize = 5;

fn brute_search(vectors: &Vec<Element>, goal: &Element) -> Vec<(usize, f32)> {
    let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();

    for (idx, &v) in vectors.iter().enumerate() {
        let d = dist(&v, goal);

        if res.len() < MAX_NEIGHBORS || d < res.peek().unwrap().0 {
            res.push((d, idx));
        }

        if res.len() > MAX_NEIGHBORS {
            res.pop();
        }
    }

    return res.into_sorted_vec().into_iter().map(|(d, idx)| (idx, d.into())).collect();
}

fn main() {
    let num_vectors = 20001;
    let (vectors, words) = file_io::read("/Users/erik/data/glove.6B/glove.6B.50d.txt", num_vectors).unwrap();

    println!("Read {} vectors", vectors.len());

    
    let mut index = hnsw::HnswBuilder::new(vectors);
    index.build_index();
    println!("Built index");
    index.write_to_disk();
    println!("Wrote to disk");

    let file = File::open("test.index").unwrap();
    let mut mmap = unsafe { Mmap::map(&file).unwrap() };

    println!("Reading");

    let (vectors, _) = file_io::read("/Users/erik/data/glove.6B/glove.6B.50d.txt", num_vectors).unwrap();

    println!("Reading index");
    let index = hnsw::Hnsw::load_from_mmap(&mut mmap, &vectors[..]);

    println!("Loaded");


    let mut pcounts = [[0; MAX_NEIGHBORS]; MAX_NEIGHBORS];

    let num_queries = 1000;
    let mut query_count = 0;
    for idx in (0..num_vectors).step_by(num_vectors / num_queries) {
        let res = index.search(&vectors[idx]);
        let brute = brute_search(&vectors, &vectors[idx]);
        query_count += 1;

        for i in 0..MAX_NEIGHBORS {
            if let Some(pos) = res.iter().take(MAX_NEIGHBORS).position(|&(x, _)| x == brute[i].0) {
                pcounts[i][pos] += 1;
            }
        }
    }

    for i in 0..MAX_NEIGHBORS {
        let mut sum = 0.0f32;

        print!("{}:\t", i);
        for j in 0..MAX_NEIGHBORS {
            sum += pcounts[i][j] as f32 / (query_count as f32);

            print!("{}\t", sum);
        }
        println!();
    }


/*
    for i in 0..LEVELS {
        for (left, right) in hnsw.levels[i].iter().zip(index.levels[i].iter()) {
            
        }
    }
*/
/*
    for i in 7000..7100 {
        println!("");
        println!("Nearest neighbors to {}:", words[i]);
        for (&(r, d), (br, bd)) in 
            index.search(&vectors[i]).iter()
            .zip(
                brute_search(&vectors, &vectors[i]).into_iter()) {
            println!("{}: {} \t {}: {}", words[r], d, words[br], bd);
        }
    }
*/

}
