#[macro_use]
extern crate clap;
extern crate granne;
extern crate time;

use clap::{App, Arg};
use std::path::Path;
use time::PreciseTime;

fn main() {
    let matches = App::new("granne::generate_query_vectors")
        .version(crate_version!())
        .about("Generate query vectors from queries")
        .arg(
            Arg::with_name("word_embeddings")
                .long("word_embeddings")
                .short("w")
                .help("Binary file with word embeddings")
                .takes_value(true)
                .required(true)
                .index(1)
        )
        .arg(
            Arg::with_name("dimension")
                .long("dimension")
                .short("d")
                .help("Word embeddings dimension")
                .takes_value(true)
                .required(true)
                .index(2)
        )
        .arg(
            Arg::with_name("queries")
                .long("queries")
                .short("q")
                .help("Binary file with query elements")
                .takes_value(true)
                .required(true)
                .index(3)
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .short("o")
                .help("Path where to write output")
                .takes_value(true)
                .index(4)
                .default_value("queries.vectors")
        ).get_matches();

    let word_embeddings_file = matches.value_of("word_embeddings").unwrap();
    let dimension: usize = matches.value_of("dimension").unwrap().parse().unwrap();
    let query_file = matches.value_of("queries").unwrap();
    let output_file = matches.value_of("output").unwrap();

    let start_time = PreciseTime::now();

    granne::query_embeddings::parsing::compute_query_vectors_and_save_to_disk(
        dimension,
        &Path::new(query_file),
        &Path::new(word_embeddings_file),
        &Path::new(output_file),
        true
    );

    let end_time = PreciseTime::now();
    let total_time = start_time.to(end_time).num_seconds();
    println!("Queries generated in {} s", total_time);
}
