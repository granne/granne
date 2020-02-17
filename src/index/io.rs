use super::*;

use serde_json;
use std::io::{BufWriter, Read, Result, Seek, SeekFrom, Write};
use std::io::{Error, ErrorKind};

const METADATA_LEN: usize = 1024;
const SERIALIZATION_VERSION: usize = 2;
const LIBRARY_STR: &str = "granne";

pub(super) fn write_index(layers: &Layers, buffer: impl Write + Seek) -> Result<()> {
    let mut buffer = BufWriter::new(buffer);

    // We would like to write metadata first, but the size of each layer needs to be
    // computed first.
    let metadata_pos = buffer.seek(SeekFrom::Current(0))?;
    buffer.seek(SeekFrom::Current(METADATA_LEN as i64))?;

    // write graph
    let num_neighbors = if layers.len() > 0 {
        layers.as_graph(layers.len() - 1).get_neighbors(0).len()
    } else {
        0
    };
    let layer_counts: Vec<usize> = (0..layers.len()).map(|i| layers.as_graph(i).len()).collect();

    let mut layer_sizes = Vec::new();
    match layers {
        Layers::FixWidth(layers) => {
            for layer in layers {
                let layer_size = layer.write_as_multi_set_vector(&mut buffer, |&x| x != UNUSED)?;
                layer_sizes.push(layer_size);
            }
        }
        Layers::Compressed(layers) => {
            for layer in layers {
                let layer_size = layer.write(&mut buffer)?;
                layer_sizes.push(layer_size);
            }
        }
    }

    // Rewind cursor and write metadata
    buffer.seek(SeekFrom::Start(metadata_pos))?;

    let mut metadata = String::with_capacity(METADATA_LEN);
    metadata.push_str(LIBRARY_STR);

    metadata.push_str(
        &serde_json::to_string(&serde_json::json!({
            "granne_version": env!("CARGO_PKG_VERSION"),
            "version": SERIALIZATION_VERSION,
            "num_elements": *layer_counts.last().unwrap_or(&0),
            "num_layers": layer_counts.len(),
            "num_neighbors": num_neighbors,
            "layer_counts": layer_counts,
            "layer_sizes": layer_sizes,
            "compressed": true,
        }))
        .expect("Could not create metadata json"),
    );

    let mut metadata = metadata.into_bytes();
    assert!(metadata.len() <= METADATA_LEN);
    metadata.resize(METADATA_LEN, b' ');

    buffer.write_all(metadata.as_slice())?;

    Ok(())
}

pub(super) fn load_layers(buffer: &'_ [u8]) -> Layers<'_> {
    let layer_sizes = read_layer_sizes(buffer).expect("Could not read metadata");

    let mut start = METADATA_LEN;

    // load graph
    let mut layers = Vec::new();
    for size in layer_sizes {
        let end = start + size;
        let layer = &buffer[start..end];
        layers.push(MultiSetVector::from_bytes(layer));
        start = end;
    }

    Layers::Compressed(layers)
}

fn read_layer_sizes<I: Read>(index_reader: I) -> Result<Vec<usize>> {
    let mut index_reader = index_reader.take(METADATA_LEN as u64);

    let mut lib_str = Vec::new();
    index_reader
        .by_ref()
        .take(LIBRARY_STR.len() as u64)
        .read_to_end(&mut lib_str)?;
    if String::from_utf8(lib_str).unwrap_or_else(|_| "".to_string()) != LIBRARY_STR {
        return Err(Error::new(ErrorKind::InvalidData, "Library string missing"));
    }

    // the current version stores metadata as json
    let metadata: serde_json::Value = serde_json::from_reader(index_reader).expect("Could not read metadata");

    let num_layers: usize = serde_json::from_value(metadata["num_layers"].clone()).expect("Could not read num_layers");
    let layer_counts: Vec<usize> =
        serde_json::from_value(metadata["layer_counts"].clone()).expect("Could not read layer_counts");
    assert_eq!(num_layers, layer_counts.len());

    let layer_sizes = &metadata["layer_sizes"];
    let layer_sizes: Vec<usize> = serde_json::from_value(layer_sizes.clone())?;

    Ok(layer_sizes)
}
