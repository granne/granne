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

extern crate granne;

use std::fs::File;
use std::io::prelude::*;

use clap::{App, Arg};

use granne::*;
use granne::file_io;

#[derive(Debug, Deserialize)]
struct Settings {
    output_file: String,
    vectors_output_file: String,
    num_layers: usize,
    max_search: usize,
    max_number_of_vectors: usize,
    compress_vectors: bool,
    scalar_input_type: String,
}


fn main() {
    let matches = App::new("granne")
        .version(crate_version!())
        .about("Builds an index for ANN searching")
        .arg(
            Arg::with_name("config_file")
                .long("config")
                .short("c")
                .help(
                    "Config file contaning configuration to use for index creation.",
                )
                .takes_value(true)
                .default_value("default_settings.toml"),
        )
        .arg(
            Arg::with_name("input_file")
                .long("input")
                .short("i")
                .help("Input file containing vectors to be indexed")
                .takes_value(true)
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("compress_vectors")
                .long("compress")
                .short("z")
                .help("Compress vectors by converting scalars into int8")
                .takes_value(false),
        )
        .get_matches();

    let config_file = matches.value_of("config_file").unwrap();
    let input_file = matches.value_of("input_file").unwrap();
    let compress_vectors = matches.is_present("compress_vectors");

    if let Ok(mut file) = File::open(config_file) {
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        if let Ok(mut settings) = toml::from_str::<Settings>(&contents) {
            if compress_vectors {
                settings.compress_vectors = compress_vectors;
            }

            println!("Using settings from {}", config_file);
            println!("{:#?}", settings);

            if settings.scalar_input_type == "int8" {
                let (vectors, _) = file_io::read(&input_file, settings.max_number_of_vectors)
                    .expect(&format!("Could not open input file: \"{}\"", input_file));

                build_and_save::<AngularIntVector<[i8; 100]>>(settings, vectors);

            } else {
                let (vectors, _) = file_io::read(&input_file, settings.max_number_of_vectors)
                    .expect(&format!("Could not open input file: \"{}\"", input_file));


                if settings.compress_vectors {
                    let vectors = vectors
                        .into_iter()
                        .map(|v: AngularVector<_>| v.into())
                        .collect();

                    build_and_save::<AngularIntVector<[i8; 100]>>(settings, vectors);

                } else {
                    build_and_save::<AngularVector<[f32; 100]>>(settings, vectors);
                }
            }

        } else {
            panic!("Malformed config file");
        }

    } else {
        panic!(
            "An error occurred: Could not open config file: {}",
            config_file
        );
    }

    return;
}


fn build_and_save<T: ComparableTo<T> + Sync + Send + Clone>(settings: Settings, vectors: Vec<T>) {

    let build_config = granne::Config {
        num_layers: settings.num_layers,
        max_search: settings.max_search,
        show_progress: true,
    };

    println!("Saving vectors to {}", settings.vectors_output_file);
    file_io::save_to_disk(&vectors[..], &settings.vectors_output_file).unwrap();

    println!("Building index...");

    let mut builder = granne::HnswBuilder::new(build_config);

    builder.add(vectors);
    builder.build_index();

    println!("Index built.");
    println!("Saving index to {}", settings.output_file);

    builder.save_to_disk(&settings.output_file).unwrap();
    println!("Completed!");
}
