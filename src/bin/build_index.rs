use clap::{crate_version, App, Arg};
use granne;
use granne::file_io;
use memmap;
use serde::Deserialize;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use time::PreciseTime;
use toml;

#[derive(Clone, Debug, Deserialize)]
struct Settings {
    /// Path to output of index file
    output_file: String,
    /// Path to output of vector file in the case of txt_input
    vectors_output_file: String,
    /// Vector dimension in case of scalar == f32
    dimension: usize,
    /// Scalar type: f32 or word_id (for query embeddings)
    scalar: String,
    /// Number of layers in index
    num_layers: usize,
    /// Search parameter during indexing
    max_search: usize,
    /// Build the index in num_build_chunks chunks and store intermediate results to output_file
    num_build_chunks: usize,
}

fn main() {
    let matches = App::new("granne")
        .version(crate_version!())
        .about("Builds an index for ANN searching")
        .arg(
            Arg::with_name("config_file")
                .long("config")
                .short("c")
                .help("Config file contaning configuration to use for index creation.")
                .takes_value(true)
                .default_value("default_settings.toml"),
        )
        .arg(
            Arg::with_name("txt_input_file")
                .long("txt_input")
                .short("t")
                .help("Text input file containing elements to be indexed")
                .takes_value(true)
                .required_unless("bin_input_file")
                .conflicts_with("bin_input_file"),
        )
        .arg(
            Arg::with_name("bin_input_file")
                .long("bin_input")
                .short("b")
                .help("Binary input file containing elements to be indexed")
                .takes_value(true)
                .required_unless("txt_input_file")
                .conflicts_with("txt_input_file"),
        )
        .arg(
            Arg::with_name("continue_index")
                .long("continue_index")
                .short("i")
                .help("Path to an already started index to continue")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("word_embeddings")
                .long("word-embeddings")
                .short("w")
                .help("Path to word embeddings file")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .short("o")
                .help("Path to index output")
                .takes_value(true)
                .required(false),
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

    let start_time = PreciseTime::now();

    if let Ok(mut file) = File::open(config_file) {
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        if let Ok(mut settings) = toml::from_str::<Settings>(&contents) {
            println!("Using settings from {}", config_file);
            if let Some(output) = matches.value_of("output") {
                println!("Overwriting output path..");
                settings.output_file = output.to_string();
            }

            println!("{:#?}", settings);

            let build_config = granne::Config {
                num_layers: settings.num_layers,
                max_search: settings.max_search,
                show_progress: true,
                num_neighbors: 20,
            };

            let mmapped_elements = {
                if is_bin_file {
                    println!("Memory mapping elements from {}", &input_file);

                    let elements = File::open(&input_file).unwrap();
                    let elements = unsafe { memmap::Mmap::map(&elements).unwrap() };

                    Some(elements)
                } else {
                    None
                }
            };

            let mmapped_elements: Option<&[u8]> = mmapped_elements.as_ref().map(|v| &v[..]);

            let existing_index = if let Some(index_path) = matches.value_of("continue_index") {
                println!("Continuing building already existing index...");

                let index_file = File::open(index_path).expect("Could not open index file");
                Some(BufReader::new(index_file))
            } else {
                println!("Building new index!");

                None
            };

            if settings.scalar == "word_id" {
                use granne::query_embeddings::QueryEmbeddings;

                assert!(is_bin_file);

                let word_embeddings_file = matches
                    .value_of("word_embeddings")
                    .expect("scalar: query requires word embeddings file");
                let word_embeddings = File::open(word_embeddings_file).expect("Could not open word embeddings file");
                let word_embeddings = unsafe { memmap::Mmap::map(&word_embeddings).unwrap() };

                let query_embeddings =
                    QueryEmbeddings::load(settings.dimension, &word_embeddings, mmapped_elements.unwrap());

                let mut builder: granne::HnswBuilder<QueryEmbeddings, granne::AngularVector>;

                if let Some(mut index_file) = existing_index {
                    builder = granne::HnswBuilder::read_index_with_borrowed_elements(
                        build_config,
                        &mut index_file,
                        &query_embeddings,
                    )
                    .expect("Could not read existing index");
                } else {
                    builder = granne::HnswBuilder::with_borrowed_elements(build_config, &query_embeddings);
                }

                build_and_save(&mut builder, settings);
            } else {
                if let Some(mmapped_elements) = mmapped_elements {
                    let elements = granne::AngularVectors::load(settings.dimension, mmapped_elements);

                    let mut builder = if let Some(mut existing_index) = existing_index {
                        granne::HnswBuilder::read_index_with_borrowed_elements(
                            build_config,
                            &mut existing_index,
                            &elements,
                        )
                        .expect("Could not read index")
                    } else {
                        granne::HnswBuilder::with_borrowed_elements(build_config, &elements)
                    };

                    build_and_save(&mut builder, settings);
                } else {
                    let mut builder = {
                        println!("Reading elements from {}", &input_file);

                        let (vector_data, _) = file_io::read_f32(&input_file).unwrap();

                        let elements = granne::AngularVectors::from_vec(settings.dimension, vector_data);

                        granne::HnswBuilder::<granne::AngularVectors, granne::AngularVector>::with_owned_elements(
                            build_config,
                            elements,
                        )
                    };

                    println!("Saving vectors to {}", settings.output_file);
                    builder.save_elements_to_disk(&settings.vectors_output_file).unwrap();

                    build_and_save(&mut builder, settings);
                }
            };

            let end_time = PreciseTime::now();
            let total_time = start_time.to(end_time).num_seconds();
            println!("Index built in {} s", total_time);
        } else {
            panic!("Malformed config file");
        }
    } else {
        panic!("An error occurred: Could not open config file: {}", config_file);
    }
}

fn build_and_save<E, T>(builder: &mut granne::HnswBuilder<E, T>, settings: Settings)
where
    E: granne::At<Output = T> + Clone + Send + Sync,
    T: granne::ComparableTo<T> + Send + Sync + Clone,
{
    if settings.num_build_chunks == 1 {
        builder.build_index();
        println!("Index built.");

        println!("Saving index to {}", settings.output_file);
        builder.save_index_to_disk(&settings.output_file).unwrap();
        println!("Completed!");
    } else {
        let chunk_size = (builder.len() + settings.num_build_chunks - 1) / settings.num_build_chunks;
        let start_chunk = 1 + (builder.indexed_elements() + chunk_size - 1) / chunk_size;

        // insert elements for all chunks except the last
        for i in start_chunk..settings.num_build_chunks {
            builder.build_index_part(i * chunk_size);

            println!("Saving index part {} to {}", i - 1, settings.output_file);
            builder.save_index_to_disk(&settings.output_file).unwrap();
        }

        // complete index by inserting the rest of the elements
        builder.build_index();
        println!("Index built.");

        println!("Saving complete index to {}", settings.output_file);
        builder.save_index_to_disk(&settings.output_file).unwrap();
        println!("Completed!");
    }
}
