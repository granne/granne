#![feature(iterator_step_by)]

extern crate arrayvec;
extern crate ordered_float;
extern crate time;
extern crate rand;
extern crate rayon;
extern crate revord;
extern crate fnv;
extern crate memmap;
extern crate hnsw;

use std::collections::BinaryHeap;
pub use ordered_float::NotNaN;
use std::fs::File;
use std::io::prelude::*;
use memmap::Mmap;
use rand::{thread_rng, Rng};

use rayon::prelude::*;

use hnsw::*;
use hnsw::file_io;

const MAX_NEIGHBORS: usize = 5;

fn brute_search(vectors: &Vec<FloatElement>, goal: &FloatElement) -> Vec<(usize, f32)> {
    let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();

    let dists: Vec<NotNaN<f32>> = vectors.par_iter().map(|v| reference_dist(v, goal)).collect();

    for (idx, d) in dists.into_iter().enumerate() {

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
    let num_vectors = 100000;
    let (mut vectors, words) = file_io::read("/Users/erik/data/glove.6B/glove.6B.100d.txt", num_vectors).unwrap();

    //thread_rng().shuffle(&mut vectors[..]);

    let vectors2 = vectors.clone();

    println!("Read {} vectors", vectors.len());

    let config = Config {
        num_levels: 6,
        level_multiplier: 12,
        max_search: 400,
        show_progress: true,
    };

    let mut index = HnswBuilder::new(config, &vectors[..]);
    index.build_index();
    println!("Built index");


    index.save_to_disk("test.index");
    println!("Wrote to disk");

    let index = index.get_index();

/*
    let file = File::open("test.index").unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    println!("Reading index");
    let index = Hnsw::load(&mmap, &vectors[..]);

    println!("Loaded");
*/
    let mut pcounts = [[0; MAX_NEIGHBORS]; MAX_NEIGHBORS];

    let max_search = 30;
    let num_queries = 1000;
    let mut query_count = 0;
    for idx in (0..num_vectors).step_by(num_vectors / num_queries) {
        let res = index.search(&vectors[idx], max_search);
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

            print!("{:.3}\t", sum);
        }
        println!();
    }


}
