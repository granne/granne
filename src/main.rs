#![feature(iterator_step_by)]

extern crate arrayvec;
extern crate ordered_float;
extern crate time;
extern crate rand;

use types::*;
use std::collections::BinaryHeap;
pub use ordered_float::NotNaN;
use rand::{thread_rng, Rng};

mod types;
mod file_io;
mod hnsw;

const MAX_NEIGHBORS: usize = 5;

fn brute_search(vectors: &Vec<Element>, goal: &Element) -> Vec<(usize, f32)> {
    let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();

    for (idx, &v) in vectors.iter().enumerate() {
        let d = NotNaN::new(dist(&v, goal)).unwrap();

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
    let mut index = hnsw::Hnsw::new(vectors);
    index.build_index();

    let (vectors, _) = file_io::read("/Users/erik/data/glove.6B/glove.6B.50d.txt", num_vectors).unwrap();

    for i in 0..100 {
        println!("");
        println!("Nearest neighbors to {}:", words[i]);
        for (&(r, d), (br, bd)) in 
            index.search(&vectors[i]).iter()
            .zip(
                brute_search(&vectors, &vectors[i]).into_iter()) {
            println!("{}: {} \t {}: {}", words[r], d, words[br], bd);
        }
    }

//             entrypoints: thread_rng().gen_iter::<usize>().map(|x| x % 5000).take(100).collect(),
//    println!("{:?}", index.search(&vectors[25512]));
//    println!("{:?}", index.search(&vectors[13]));
//    println!("{:?}", index.search(&vectors[0]));
//    println!("{:?}", index.search(&vectors[112000]));


}
