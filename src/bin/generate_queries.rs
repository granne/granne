#[macro_use]
extern crate clap;
extern crate granne;
extern crate time;

use clap::{App, Arg};
use std::path::Path;
use time::PreciseTime;

fn main() {
    let matches = App::new("granne::generate_queries")
        .version(crate_version!())
        .about("Generate compact queries")
        .arg(
            Arg::with_name("words")
                .long("words")
                .short("w")
                .help("Text file containing one word per line")
                .takes_value(true)
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("queries")
                .long("queries")
                .short("q")
                .help("Text file or directory with text files containing one json query string per line")
                .takes_value(true)
                .required(true)
                .index(2)
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .short("o")
                .help("Path for output")
                .takes_value(true)
                .index(3)
                .default_value("queries.bin")
        ).get_matches();

    let word_path = matches.value_of("words").unwrap();
    let query_path = matches.value_of("queries").unwrap();
    let output_path = matches.value_of("output").unwrap();

    let start_time = PreciseTime::now();
    println!("Reading queries from {:?} and generating queries...", query_path);

    granne::query_embeddings::parsing::parse_queries_and_save_to_disk(
        &Path::new(query_path),
        &Path::new(word_path),
        &Path::new(output_path),
        true
    );

    let end_time = PreciseTime::now();
    let total_time = start_time.to(end_time).num_seconds();
    println!("Queries generated in {} s", total_time);
}
