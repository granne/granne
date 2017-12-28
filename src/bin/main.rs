extern crate arrayvec;
extern crate ordered_float;
extern crate time;
extern crate rand;
extern crate rayon;
extern crate revord;
extern crate fnv;
extern crate memmap;
extern crate num_iter;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate serde_derive;
extern crate toml;

extern crate hnsw;

use std::collections::BinaryHeap;
pub use ordered_float::NotNaN;
use std::fs::File;
use std::io::prelude::*;
use memmap::Mmap;
use rand::{thread_rng, Rng};

use rayon::prelude::*;

use clap::{App, Arg};

use hnsw::*;
use hnsw::file_io;

const MAX_NEIGHBORS: usize = 5;

fn brute_search<T: HasDistance + Sync + Send>(vectors: &[T], goal: &T) -> Vec<(usize, f32)> {
    let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();

    let dists: Vec<NotNaN<f32>> = vectors.par_iter().map(|v| v.dist(goal)).collect();

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

#[derive(Debug, Deserialize)]
struct Settings {
    output_file: String,
    vectors_output_file: String,
    num_layers: usize,
    max_search: usize,
    max_number_of_vectors: usize,
    compress_vectors: bool,
}


fn main() {
    let matches =
        App::new("Hnsw")
        .version(crate_version!())
        .about("Builds an index for ANN searching")
        .arg(Arg::with_name("config_file")
             .long("config")
             .short("c")
             .help("Config file contaning configuration to use for index creation.")
             .takes_value(true)
             .default_value("default_settings.toml"))
        .arg(Arg::with_name("input_file")
             .long("input")
             .short("i")
             .help("Input file containing vectors to be indexed")
             .takes_value(true)
             .required(true)
             .index(1))
        .arg(Arg::with_name("compress_vectors")
             .long("compress")
             .short("z")
             .help("Compress vectors by converting scalars into int8")
             .takes_value(false))
        .get_matches();

    let config_file = matches.value_of("config_file").unwrap();
    let input_file = matches.value_of("input_file").unwrap();
    let compress_vectors = matches.is_present("compress_vectors");

    if let Ok(mut file) = File::open(config_file) {
        let mut contents = String::new();
        file.read_to_string(&mut contents);

        if let Ok(mut settings) = toml::from_str::<Settings>(&contents) {
            if compress_vectors {
                settings.compress_vectors = compress_vectors;
            }

            println!("Using settings from {}", config_file);
            println!("{:#?}", settings);

            build_and_save(settings, &input_file);

        } else {
            panic!("Malformed config file");
        }

    } else {
        panic!("An error occurred: Could not open config file: {}", config_file);
    }

    return;
}

fn build_and_save(settings: Settings, input_file: &str) {

    let (vectors, _) =
        file_io::read(
            &input_file,
            settings.max_number_of_vectors
        ).expect(&format!("Could not open input file: \"{}\"",
                          input_file));

    println!("{} vectors read from {}", vectors.len(), input_file);

    let mut vectors: Vec<_> =
        vectors.into_iter().map(|v| v.normalized()).collect();

    thread_rng().shuffle(&mut vectors[..]);


    let build_config = hnsw::Config {
        num_layers: settings.num_layers,
        max_search: settings.max_search,
        show_progress: true
    };

    if settings.compress_vectors {
        let vectors: Vec<Int8Element> =
            vectors.into_iter().map(|v| v.into()).collect();

        println!("Saving vectors to {}", settings.vectors_output_file);
        file_io::save_to_disk(&vectors[..], &settings.vectors_output_file);

        println!("Building index...");

        let mut builder = hnsw::HnswBuilder::new(build_config);

        builder.add(vectors);
        builder.build_index();

        println!("Index built.");
        println!("Saving index to {}", settings.output_file);

        builder.save_to_disk(&settings.output_file);

        test_index::<Int8Element>();

    } else {

        println!("Saving vectors to {}", settings.vectors_output_file);
        file_io::save_to_disk(&vectors[..], &settings.vectors_output_file);

        println!("Building index...");

        let mut builder = hnsw::HnswBuilder::new(build_config);

        builder.add(vectors);
        builder.build_index();

        println!("Index built.");
        println!("Saving index to {}", settings.output_file);

        builder.save_to_disk(&settings.output_file);

        test_index::<NormalizedFloatElement>();
    }
}


fn test_index<T: HasDistance + Sync + Send + Clone>() {

    let config_file = "default_settings.toml";

    if let Ok(mut file) = File::open(config_file) {
        let mut contents = String::new();
        file.read_to_string(&mut contents);

        if let Ok(mut settings) = toml::from_str::<Settings>(&contents) {

            println!("Loading index...");

            let file = File::open(settings.output_file).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };
            let index = Hnsw::load(&mmap);
            println!("Loaded");

            println!("Loading vectors...");
            let vectors: Vec<T> =
                file_io::load_from_disk(&settings.vectors_output_file).unwrap();

            println!("Loaded");

            let num_vectors = vectors.len();
            assert_eq!(num_vectors, index.len());

            let mut pcounts = [[0; MAX_NEIGHBORS]; MAX_NEIGHBORS];

            let max_search = 20;
            let num_queries = 2500;
            let mut query_count = 0;
            for idx in num_iter::range_step(0, num_vectors, num_vectors / num_queries) {
                let res = index.search(&vectors[idx], MAX_NEIGHBORS, max_search);
                let brute = brute_search(&vectors, &vectors[idx]);
                query_count += 1;

                for i in 0..MAX_NEIGHBORS {
                    if let Some(pos) = res.iter().position(|&(x, _)| x == brute[i].0) {
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
    }


}
