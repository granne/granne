use super::*;
use byteorder::{LittleEndian, ReadBytesExt};
use memmap::Mmap;
use serde_json;
use std::io::{BufReader, BufWriter, Read, Result, Seek, SeekFrom, Write};

pub fn save_index_to_disk(layers: &[FixedWidthSliceVector<NeighborId>], file: &mut File, compress: bool) -> Result<()> {
    let mut file = BufWriter::new(file);

    // We would like to write metadata first, but the size of each layer needs to be
    // computed first.
    let metadata_pos = file.seek(SeekFrom::Current(0))?;
    file.seek(SeekFrom::Current(METADATA_LEN as i64))?;

    let mut layer_sizes = Vec::new();
    let (num_neighbors, num_elements) = if !layers.is_empty() {
        (layers[0].get(0).len(), layers.last().unwrap().len())
    } else {
        (0, 0)
    };

    // write graph
    if compress {
        for layer in layers {
            let layer_size =
                layer.write_as_variable_width_slice_vector::<NeighborId, _, _>(&mut file, |&x| x != UNUSED)?;
            layer_sizes.push(layer_size);
        }
    } else {
        for layer in layers {
            let layer_size = layer.len() * num_neighbors * std::mem::size_of::<NeighborId>();
            layer_sizes.push(layer_size);
            layer.write(&mut file)?;
        }
    }

    // Rewind cursor and write metadata
    file.seek(SeekFrom::Start(metadata_pos))?;

    let mut metadata = String::with_capacity(METADATA_LEN);
    metadata.push_str(LIBRARY_STR);

    metadata.push_str(
        &serde_json::to_string(&json!({
            "version": SERIALIZATION_VERSION,
            "num_elements": num_elements,
            "num_layers": layers.len(),
            "num_neighbors": num_neighbors,
            "layer_counts": layers.iter().map(|layer| layer.len()).collect::<Vec<_>>(),
            "layer_sizes": layer_sizes,
            "compressed": compress
        }))
        .expect("Could not create metadata json"),
    );

    let mut metadata = metadata.into_bytes();
    assert!(metadata.len() <= METADATA_LEN);
    metadata.resize(METADATA_LEN, ' ' as u8);

    file.write_all(metadata.as_slice())?;

    Ok(())
}

pub fn compress_index(input: &str, output: &str) -> Result<()> {
    let index = File::open(input)?;
    let index = unsafe { Mmap::map(&index)? };

    if let Layers::Standard(layers) = load_layers(&index[..]) {
        let mut file = File::create(output)?;
        save_index_to_disk(&layers, &mut file, true)
    } else {
        panic!("Index already compressed");
    }
}

pub fn load_layers<'a>(buffer: &'a [u8]) -> Layers<'a> {
    let (num_neighbors, layer_sizes, compressed) = read_metadata(buffer).expect("Could not read metadata");

    let mut start = METADATA_LEN;

    // load graph
    if compressed {
        let mut layers = Vec::new();
        for size in layer_sizes {
            let end = start + size;
            let layer = &buffer[start..end];
            layers.push(VariableWidthSliceVector::load(layer));
            start = end;
        }

        Layers::Compressed(layers)
    } else {
        let mut layers = Vec::new();
        for size in layer_sizes {
            let end = start + size;
            let layer = &buffer[start..end];
            layers.push(FixedWidthSliceVector::load(layer, num_neighbors));
            start = end;
        }

        Layers::Standard(layers)
    }
}

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
    let (last_layer_idx, last_layer_count) = if let Some((num_layers, last_layer_count)) = num_layers_and_count {
        (num_layers - 1, last_layer_count)
    } else {
        (<usize>::max_value(), 0)
    };

    for (layer_idx, layer_size) in layer_sizes.into_iter().enumerate() {
        let layer_reader = index_reader.by_ref().take(layer_size as u64);

        let layer = if layer_idx != last_layer_idx {
            FixedWidthSliceVector::read(layer_reader, num_neighbors)?
        } else {
            FixedWidthSliceVector::read_with_capacity(layer_reader, num_neighbors, last_layer_count)?
        };

        assert_eq!(layer_size / node_size, layer.len());

        layers.push(layer);
    }

    Ok(layers)
}

fn read_metadata<I: Read>(index_reader: I) -> Result<(usize, Vec<usize>, bool)> {
    let mut index_reader = index_reader.take(METADATA_LEN as u64);

    // Check if the file is a current or old version of an granne index
    let mut lib_str = Vec::new();
    index_reader
        .by_ref()
        .take(LIBRARY_STR.len() as u64)
        .read_to_end(&mut lib_str)?;
    if String::from_utf8(lib_str).unwrap_or("".to_string()) == LIBRARY_STR {
        // the current version stores metadata as json
        let metadata: serde_json::Value = serde_json::from_reader(index_reader).expect("Could not read metadata");

        let num_neighbors: usize =
            serde_json::from_value(metadata["num_neighbors"].clone()).expect("Could not read num_neighbors");
        let num_layers: usize =
            serde_json::from_value(metadata["num_layers"].clone()).expect("Could not read num_layers");
        let layer_counts: Vec<usize> =
            serde_json::from_value(metadata["layer_counts"].clone()).expect("Could not read layer_counts");
        assert_eq!(num_layers, layer_counts.len());

        let compressed: bool = serde_json::from_value(metadata["compressed"].clone()).unwrap_or(false);

        let layer_sizes: Vec<usize> = if let Some(layer_sizes) = metadata.get("layer_sizes") {
            serde_json::from_value(layer_sizes.clone()).unwrap()
        } else {
            assert!(!compressed);

            layer_counts
                .iter()
                .map(|c| c * num_neighbors * std::mem::size_of::<NeighborId>())
                .collect()
        };

        Ok((num_neighbors, layer_sizes, compressed))
    } else {
        // Read legacy index
        // First 8 bytes are num_nodes. We can ignore these.
        // Note that "granne".len() bytes were already read.
        const LIBRARY_STR_LEN: usize = 6;
        assert_eq!(LIBRARY_STR_LEN, LIBRARY_STR.len());
        const BYTES_LEFT_FOR_NUM_NODES: usize = std::mem::size_of::<usize>() - LIBRARY_STR_LEN;

        index_reader.read_exact(&mut [0; BYTES_LEFT_FOR_NUM_NODES])?;

        let num_layers = index_reader.read_u64::<LittleEndian>()? as usize;
        let num_neighbors = 20;

        let mut layer_counts = Vec::new();
        let mut layer_sizes = Vec::new();
        for _ in 0..num_layers {
            let count = index_reader.read_u64::<LittleEndian>()? as usize;
            layer_counts.push(count);
            layer_sizes.push(count * num_neighbors * std::mem::size_of::<NeighborId>())
        }

        let compressed = false;

        Ok((num_neighbors, layer_sizes, compressed))
    }
}
