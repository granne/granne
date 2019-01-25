use super::*;

use crate::types::example::*;
use crate::types::*;

#[test]
fn neighbor_id_conversions() {
    for &integer in &[
        0usize,
        1usize,
        2usize,
        3usize,
        1234567usize,
        NeighborId::max_value() - 1,
        NeighborId::max_value(),
    ] {
        let neighbor_id: NeighborId = integer.into();
        let into_integer: usize = neighbor_id.into();
        assert_eq!(integer, into_integer);
    }
}

#[test]
fn neighbor_id_cast() {
    let integer: usize = 3301010345;
    let neighbor_id: NeighborId = integer.into();

    let reinterpreted = &neighbor_id.0[0] as *const u8 as *const u32;
    let reinterpreted = unsafe { *reinterpreted };

    assert_eq!(integer, reinterpreted as usize);
}

#[test]
fn select_neighbors() {
    const DIM: usize = 50;
    let element: AngularVector = random_dense_element(DIM);

    let other_elements: Vec<AngularVector> = (0..50).map(|_| random_dense_element(DIM)).collect();

    let mut candidates: Vec<_> = other_elements.iter().map(|e| e.dist(&element)).enumerate().collect();

    candidates.sort_unstable_by_key(|&(_, d)| d);

    let neighbors = HnswBuilder::select_neighbors(&other_elements[..], candidates.clone(), 10);

    assert!(0 < neighbors.len() && neighbors.len() <= 10);

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }

    let neighbors = HnswBuilder::select_neighbors(&other_elements[..], candidates.clone(), 60);

    assert_eq!(candidates.len(), neighbors.len());

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }
}

fn build_and_search<T: ComparableTo<T> + Sync + Send + Clone>(elements: Vec<T>) {
    let config = Config {
        num_layers: 5,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let mut builder = HnswBuilder::<[T], T>::with_owned_elements(config, elements);
    builder.build_index();
    let index = builder.get_index();

    verify_search(&index, 0.95, 40);
}

fn verify_search<E: At<Output = T> + ?Sized, T: ComparableTo<T> + Sync + Send + Clone>(
    index: &Hnsw<E, T>,
    precision: f32,
    max_search: usize,
) {
    let mut num_found = 0;
    for i in 0..index.len() {
        if index.search(&index.get_element(i), 1, max_search)[0].0 == i {
            num_found += 1;
        }
    }

    let p1 = num_found as f32 / index.len() as f32;

    println!("p1: {}", p1);
    assert!(precision < p1);
}

#[test]
fn with_borrowed_elements() {
    let config = Config {
        num_layers: 5,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let elements: Vec<_> = (0..500).map(|_| random_dense_element::<AngularVector>(25)).collect();

    let mut builder = HnswBuilder::with_borrowed_elements(config, elements.as_slice());

    assert_eq!(elements.len(), builder.elements.len());
    builder.build_index();
    let index = builder.get_index();

    assert_eq!(index.len(), elements.len());

    verify_search(&index, 0.95, 40);
}

#[test]
fn with_elements_and_add() {
    let config = Config {
        num_layers: 5,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let elements: AngularVectors = (0..500).map(|_| random_dense_element::<AngularVector>(25)).collect();
    let additional_elements: AngularVectors = (0..100).map(|_| random_dense_element::<AngularVector>(25)).collect();

    let mut builder: HnswBuilder<AngularVectors, AngularVector> = HnswBuilder::with_owned_elements(config, elements);

    assert_eq!(500, builder.elements.len());

    builder.add(additional_elements);

    assert_eq!(600, builder.elements.len());

    builder.build_index();

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn build_and_search_float() {
    let elements: Vec<_> = (0..1500).map(|_| random_dense_element::<AngularVector>(128)).collect();

    build_and_search(elements);
}

#[test]
fn build_and_search_int8() {
    const DIM: usize = 32;

    let elements: Vec<AngularIntVector> = (0..500)
        .map(|_| random_dense_element::<AngularVector>(DIM).into())
        .collect();

    build_and_search(elements);
}

#[test]
fn incremental_build_0() {
    let elements: Vec<_> = (0..1000).map(|_| random_dense_element::<AngularVector>(25)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let layer_multiplier = compute_layer_multiplier(elements.len(), config.num_layers);

    let mut builder = HnswBuilder::with_borrowed_elements(config, elements.as_slice());

    builder.build_index_part(layer_multiplier + 2);

    assert_eq!(3, builder.layers.len());
    assert_eq!(layer_multiplier.pow(0), builder.layers[0].len());
    assert_eq!(layer_multiplier.pow(1), builder.layers[1].len());
    assert_eq!(layer_multiplier + 2, builder.layers[2].len());

    builder.build_index_part(layer_multiplier.pow(2) + 2);

    assert_eq!(4, builder.layers.len());
    assert_eq!(layer_multiplier.pow(0), builder.layers[0].len());
    assert_eq!(layer_multiplier.pow(1), builder.layers[1].len());
    assert_eq!(layer_multiplier.pow(2), builder.layers[2].len());
    assert_eq!(layer_multiplier.pow(2) + 2, builder.layers[3].len());

    builder.build_index_part(elements.len());

    assert_eq!(4, builder.layers.len());
    assert_eq!(layer_multiplier.pow(0), builder.layers[0].len());
    assert_eq!(layer_multiplier.pow(1), builder.layers[1].len());
    assert_eq!(layer_multiplier.pow(2), builder.layers[2].len());
    assert_eq!(elements.len(), builder.layers[3].len());

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn incremental_build_1() {
    let elements: Vec<_> = (0..1000).map(|_| random_dense_element::<AngularVector>(25)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let mut builder = HnswBuilder::with_borrowed_elements(config.clone(), elements.as_slice());

    let num_chunks = 10;
    let chunk_size = elements.len() / num_chunks;
    assert_eq!(elements.len(), num_chunks * chunk_size);

    for i in 1..(num_chunks + 1) {
        builder.build_index_part(i * chunk_size);

        assert_eq!(i * chunk_size, builder.indexed_elements());
    }

    assert_eq!(config.num_layers, builder.layers.len());
    assert_eq!(elements.len(), builder.indexed_elements());

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn incremental_build_with_write_and_read() {
    const DIM: usize = 25;

    let elements: AngularVectors = (0..1000).map(|_| random_dense_element::<AngularVector>(DIM)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let num_chunks = 10;
    let chunk_size = elements.len() / num_chunks;
    assert_eq!(elements.len(), num_chunks * chunk_size);

    let mut data = Vec::new();
    {
        let builder = HnswBuilder::with_borrowed_elements(config.clone(), &elements);
        builder.write(&mut data).unwrap();
    }

    for i in 0..num_chunks {
        let mut builder =
            HnswBuilder::read_index_with_borrowed_elements(config.clone(), &mut data.as_slice(), &elements).unwrap();
        assert_eq!(i * chunk_size, builder.indexed_elements());

        builder.build_index_part((i + 1) * chunk_size);

        assert_eq!((i + 1) * chunk_size, builder.indexed_elements());

        data.clear();
        builder.write(&mut data).unwrap();
    }

    let index = Hnsw::<AngularVectors, AngularVector>::load(data.as_slice(), &elements);
    assert_eq!(config.num_layers, index.layers.len());
    assert_eq!(elements.len(), index.len());

    verify_search(&index, 0.95, 40);
}

#[test]
fn read_index_with_owned_elements() {
    let num_elements = 1000;
    const DIM: usize = 25;
    type Element = AngularVector<'static>;

    let mut owning_builder = {
        let elements: Vec<Element> = (0..num_elements)
            .map(|_| random_dense_element::<Element>(DIM))
            .collect();

        let config = Config {
            num_layers: 4,
            num_neighbors: 20,
            max_search: 50,
            show_progress: false,
        };

        let mut builder = HnswBuilder::with_borrowed_elements(config.clone(), elements.as_slice());
        builder.build_index_part(num_elements / 2);
        let mut data = Vec::new();
        builder.write(&mut data).unwrap();

        let owning_builder: HnswBuilder<[Element], Element> =
            HnswBuilder::read_index_with_owned_elements(config.clone(), &mut data.as_slice(), elements.clone())
                .unwrap();

        assert_eq!(num_elements / 2, builder.indexed_elements());
        assert_eq!(num_elements / 2, owning_builder.indexed_elements());
        assert_eq!(num_elements, builder.len());
        assert_eq!(num_elements, owning_builder.len());

        assert_eq!(builder.layers.len(), owning_builder.layers.len());

        owning_builder
    };

    // owning_builder still alive after elements go out of scope
    assert_eq!(num_elements / 2, owning_builder.indexed_elements());

    owning_builder.build_index();

    assert_eq!(num_elements, owning_builder.indexed_elements());
    assert_eq!(num_elements, owning_builder.len());
}

#[test]
fn read_legacy_index() {
    let file = File::open("example_data/legacy_index.granne").expect("Could not open file with legacy index");

    let layers = read_layers(file, None).expect("Could not read index");

    assert_eq!(8, layers.len());
    assert_eq!(100, layers.last().unwrap().len());
    assert_eq!(20 + EXTRA_NEIGHBORS_AT_BUILD_TIME, layers.last().unwrap().get(0).len());
}

#[test]
fn empty_build() {
    let elements: AngularVectors = (0..100).map(|_| random_dense_element::<AngularVector>(25)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    let mut builder = HnswBuilder::with_borrowed_elements(config.clone(), &elements);

    builder.build_index_part(0);
}

#[test]
fn test_layer_multiplier() {
    assert_eq!(2, compute_layer_multiplier(10, 5));
    assert_eq!(14, compute_layer_multiplier(400000, 6));
    assert_eq!(22, compute_layer_multiplier(2000000000, 8));
    assert_eq!(555, compute_layer_multiplier(555, 2));
    assert_eq!(25, compute_layer_multiplier(625, 3));
}

#[test]
fn write_and_load() {
    const DIM: usize = 50;
    let elements: AngularVectors = (0..100).map(|_| random_dense_element(DIM)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 10,
        show_progress: false,
    };

    let mut builder = HnswBuilder::with_borrowed_elements(config, &elements);
    builder.build_index();

    let mut data = Vec::new();
    builder.write(&mut data).unwrap();

    let index = Hnsw::<AngularVectors, AngularVector>::load(&data[..], &elements);

    assert_eq!(builder.layers.len(), index.layers.len());

    for layer in 0..builder.layers.len() {
        assert_eq!(builder.layers[layer].len(), index.layers[layer].len());

        for i in 0..builder.layers[layer].len() {
            let builder_neighbors: Vec<_> = builder.layers[layer]
                .get(i)
                .iter()
                .take_while(|&&n| n != UNUSED)
                .collect();
            let index_neighbors: Vec<_> = index.layers[layer]
                .get(i)
                .iter()
                .take_while(|&&n| n != UNUSED)
                .collect();

            assert_eq!(builder_neighbors, index_neighbors);
        }
    }

    assert_eq!(builder.elements.len(), index.len());

    for i in 0..builder.elements.len() {
        assert!(builder.elements.at(i).dist(&index.get_element(i)).into_inner() < DIST_EPSILON);
    }
}

#[test]
fn write_and_read() {
    const DIM: usize = 64;

    let elements: AngularIntVectors = (0..100)
        .map(|_| random_dense_element::<AngularVector>(DIM).into())
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 10,
        show_progress: false,
    };

    let mut original = HnswBuilder::with_borrowed_elements(config.clone(), &elements);
    original.build_index();

    let mut data = Vec::new();
    original.write(&mut data).unwrap();

    let copy = HnswBuilder::<AngularIntVectors, AngularIntVector>::read_index_with_borrowed_elements(
        config,
        &mut data.as_slice(),
        &elements,
    )
    .unwrap();

    assert_eq!(original.layers.len(), copy.layers.len());

    for layer in 0..original.layers.len() {
        assert_eq!(original.layers[layer].len(), copy.layers[layer].len());

        for i in 0..original.layers[layer].len() {
            let original_neighbors: Vec<_> = original.layers[layer]
                .get(i)
                .iter()
                .take_while(|&&n| n != UNUSED)
                .collect();
            let copy_neighbors: Vec<_> = copy.layers[layer].get(i).iter().take_while(|&&n| n != UNUSED).collect();

            assert_eq!(original_neighbors, copy_neighbors);
        }
    }

    assert_eq!(original.elements.len(), copy.elements.len());
}

#[test]
fn append_elements() {
    let elements: AngularVectors = (0..500).map(|_| random_dense_element::<AngularVector>(50)).collect();

    let additional_elements: AngularVectors = (0..500).map(|_| random_dense_element::<AngularVector>(50)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        show_progress: false,
    };

    // insert half of the elements
    let mut builder = HnswBuilder::new(50, config);
    builder.add(elements.clone());
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(500, builder.layers[3].len());

    let max_search = 50;

    // assert that one arbitrary element is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements.at(123), 1, max_search)
            .iter()
            .any(|&(idx, _)| 123 == idx,));
    }

    // insert rest of the elements
    builder.add(additional_elements.clone());
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(1000, builder.layers[3].len());

    // assert that the same arbitrary element and a newly added one
    // is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements.at(123), 1, max_search)
            .iter()
            .any(|&(idx, _)| 123 == idx,));

        assert!(index
            .search(&additional_elements.at(123), 1, max_search)
            .iter()
            .any(|&(idx, _)| elements.len() + 123 == idx,));
    }
}
