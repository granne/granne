use super::*;

use serde_json;
use std::io::{BufReader, BufWriter, Read, Result, Seek, SeekFrom, Write};
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
    let layer_counts: Vec<usize> = (0..layers.len())
        .map(|i| layers.as_graph(i).len())
        .collect();

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
/*
pub fn compress_index(input: &str, output: &str) -> Result<()> {
    let index = File::open(input)?;
    let index = unsafe { Mmap::map(&index)? };

    let layers = load_layers(&index[..]);
    let mut file = File::create(output)?;

    save_index_to_disk(&layers, &mut file, true)
}
*/
pub(super) fn load_layers(buffer: &'_ [u8]) -> Layers<'_> {
    let (num_neighbors, layer_sizes, compressed) =
        read_metadata(buffer).expect("Could not read metadata");

    let mut start = METADATA_LEN;

    // load graph
    if compressed {
        let mut layers = Vec::new();
        for size in layer_sizes {
            let end = start + size;
            let layer = &buffer[start..end];
            layers.push(MultiSetVector::from_bytes(layer));
            start = end;
        }

        Layers::Compressed(layers)
    } else {
        panic!()
    }
}
/*
pub fn read_layers(
    buffer: &mut [u8],
    num_neighbors: usize,
) -> Vec<FixedWidthSliceVector<'static, NeighborId>> {
    match io::load_layers(buffer) {
        Layers::FixWidth(layers) => layers.into_iter().map(|l| l.into_owned()).collect(),
        Layers::VarWidth(layers) => {
            let fix_width_layers = Vec::new();

            for layer in layers {
                let mut fix_width_layer = FixedWidthSliceVector::with_width(num_neighbors);
                fix_width_layer.resize(layer.len(), UNUSED);

                for (i, node) in layer.iter().enumerate() {
                    // clone at most num_neighbors neighbors
                    let n = cmp::min(num_neighbors, node.len());

                    fix_width_layer.get_mut(i)[..n].clone_from_slice(&node[..n]);
                }
            }

            fix_width_layers
        }
    }
}
*/
/*
pub fn read_layers<I: Read>(
    index_reader: I,
    num_layers_and_count: Option<(usize, usize)>,
) -> Result<Vec<FixedWidthSliceVector<'static, NeighborId>>> {
    use std::mem::size_of;

    let mut index_reader = BufReader::new(index_reader);

    let (num_neighbors, layer_sizes, compressed) = read_metadata(index_reader.by_ref())?;

    assert!(!compressed, "Cannot read compressed index");

    // read graph
    let mut layers = Vec::new();
    let node_size = num_neighbors * size_of::<NeighborId>();

    // if last layer idx and size was passed in, we use this to allocate the full layer before reading
    let (last_layer_idx, last_layer_count) =
        if let Some((num_layers, last_layer_count)) = num_layers_and_count {
            (num_layers - 1, last_layer_count)
        } else {
            (<usize>::max_value(), 0)
        };

    for (layer_idx, layer_size) in layer_sizes.into_iter().enumerate() {
        let layer_reader = index_reader.by_ref().take(layer_size as u64);

        let layer = if layer_idx != last_layer_idx {
            FixedWidthSliceVector::read(layer_reader, num_neighbors)?
        } else {
            FixedWidthSliceVector::read_with_capacity(
                layer_reader,
                num_neighbors,
                last_layer_count,
            )?
        };

        assert_eq!(layer_size / node_size, layer.len());

        layers.push(layer);
    }

    Ok(layers)
}
*/
fn read_metadata<I: Read>(index_reader: I) -> Result<(usize, Vec<usize>, bool)> {
    let mut index_reader = index_reader.take(METADATA_LEN as u64);

    // Check if the file is a current or old version of an granne index
    let mut lib_str = Vec::new();
    index_reader
        .by_ref()
        .take(LIBRARY_STR.len() as u64)
        .read_to_end(&mut lib_str)?;
    if String::from_utf8(lib_str).unwrap_or_else(|_| "".to_string()) != LIBRARY_STR {
        return Err(Error::new(ErrorKind::InvalidData, "Library string missing"));
    }

    // the current version stores metadata as json
    let metadata: serde_json::Value =
        serde_json::from_reader(index_reader).expect("Could not read metadata");

    let num_neighbors: usize = serde_json::from_value(metadata["num_neighbors"].clone())
        .expect("Could not read num_neighbors");
    let num_layers: usize =
        serde_json::from_value(metadata["num_layers"].clone()).expect("Could not read num_layers");
    let layer_counts: Vec<usize> = serde_json::from_value(metadata["layer_counts"].clone())
        .expect("Could not read layer_counts");
    assert_eq!(num_layers, layer_counts.len());

    let version: usize = serde_json::from_value(metadata["version"].clone()).unwrap_or(0);

    let mut compressed: bool =
        serde_json::from_value(metadata["compressed"].clone()).unwrap_or(version >= 2);

    if compressed && version < 2 {
        compressed = false;
    }

    let layer_sizes = &metadata["layer_sizes"];
    let layer_sizes: Vec<usize> = serde_json::from_value(layer_sizes.clone())?;

    Ok((num_neighbors, layer_sizes, compressed))
}
