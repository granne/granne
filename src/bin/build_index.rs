extern crate memmap;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate serde_derive;
extern crate toml;

extern crate granne;

use std::fs::File;
use std::io::prelude::*;

use clap::{App, Arg};

use granne::file_io;

#[derive(Debug, Deserialize)]
struct Settings {
    output_file: String,
    vectors_output_file: String,
    dimension: usize,
    scalar: String, // f32
    num_layers: usize,
    max_search: usize,
    max_number_of_vectors: usize,
    compress_vectors: bool, // unused
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
            Arg::with_name("txt_input_file")
                .long("txt_input")
                .short("i")
                .help("Text input file containing elements to be indexed")
                .takes_value(true)
                .required_unless("bin_input_file")
                .conflicts_with("bin_input_file")
        )
        .arg(
            Arg::with_name("bin_input_file")
                .long("bin_input")
                .short("b")
                .help("Binary input file containing elements to be indexed")
                .takes_value(true)
                .required_unless("txt_input_file")
                .conflicts_with("txt_input_file")
        )
        .get_matches();

    let config_file = matches.value_of("config_file").unwrap();
    let (is_bin_file, input_file) = {
        if let Some(input_file) = matches.value_of("txt_input_file") {
            (false, input_file)
        } else {
            (true, matches.value_of("bin_input_file").unwrap())
        }
    };

    if let Ok(mut file) = File::open(config_file) {
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        if let Ok(mut settings) = toml::from_str::<Settings>(&contents) {

            println!("Using settings from {}", config_file);
            println!("{:#?}", settings);

            let build_config = granne::Config {
                num_layers: settings.num_layers,
                max_search: settings.max_search,
                show_progress: true,
            };

            let mmapped_elements = {
                if is_bin_file {

                    println!("Memory mapping elements from {}", &input_file);

                    let elements = File::open("test.bin").unwrap();
                    let elements = unsafe { memmap::Mmap::map(&elements).unwrap() };

                    Some(elements)

                } else {
                    None
                }
            };

            let mmapped_elements: Option<&[u8]> =
                mmapped_elements.as_ref().map(|v| &v[..]);

            let mut builder = granne::boxed_index_builder(
                settings.dimension, build_config, mmapped_elements);

            if mmapped_elements.is_none() {
                println!("Reading elements from {}", &input_file);

                let (vectors, _) = file_io::read(&input_file,
                                                 settings.max_number_of_vectors).unwrap();

                for vec in vectors {
                    builder.add(vec);
                }

                println!("Saving vectors to {}", settings.output_file);
                builder.save_elements_to_disk(&settings.vectors_output_file).unwrap();
            }

            println!("Building index...");
            builder.build();
            println!("Index built.");

            println!("Saving index to {}", settings.output_file);
            builder.save_index_to_disk(&settings.output_file).unwrap();
            println!("Completed!");

        } else {
            panic!("Malformed config file");
        }

    } else {
        panic!(
            "An error occurred: Could not open config file: {}",
            config_file
        );
    }

}
