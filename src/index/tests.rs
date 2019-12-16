use super::*;

use crate::{angular, angular_int, test_helper, Dist};

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use tempfile;

const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

#[test]
fn select_neighbors() {
    const DIM: usize = 50;
    let element: angular::Vector = test_helper::random_vector(DIM);

    let other_elements: Vec<angular::Vector> =
        (0..50).map(|_| test_helper::random_vector(DIM)).collect();

    let mut candidates: Vec<_> = other_elements
        .iter()
        .map(|e| e.dist(&element))
        .enumerate()
        .collect();

    candidates.sort_unstable_by_key(|&(_, d)| d);

    let neighbors =
        GranneBuilder::select_neighbors(&other_elements.as_slice(), candidates.clone(), 10);

    assert!(0 < neighbors.len() && neighbors.len() <= 10);

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }

    let neighbors =
        GranneBuilder::select_neighbors(&other_elements.as_slice(), candidates.clone(), 60);

    assert_eq!(candidates.len(), neighbors.len());

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }
}

fn build_and_search<Elements: ElementContainer + Sync + Send + Clone>(elements: Elements) {
    let config = Config {
        num_layers: 5,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut builder = GranneBuilder::new(config, elements);
    builder.build_index();
    let index = builder.get_index();

    verify_search(&index, 0.95, 40);
}

fn verify_search<Elements: ElementContainer>(
    index: &Granne<Elements>,
    precision: f32,
    max_search: usize,
) {
    let mut num_found = 0;
    for i in 0..index.len() {
        if index.search(&index.get_element(i), max_search, 1)[0].0 == i {
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
        reinsert_elements: true,
        show_progress: false,
    };

    let elements: Vec<_> = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let mut builder = GranneBuilder::new(config, elements.as_slice());

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
        reinsert_elements: true,
        show_progress: false,
    };

    let elements: angular::Vectors = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();
    let additional_elements: Vec<angular::Vector> = (0..100)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let mut builder = GranneBuilder::new(config, elements);

    assert_eq!(500, builder.elements.len());

    for element in additional_elements {
        builder.push(element);
    }

    assert_eq!(600, builder.elements.len());

    builder.build_index();

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn build_and_search_float() {
    let elements: Vec<_> = (0..1500)
        .map(|_| test_helper::random_vector::<angular::Vector>(128))
        .collect();

    build_and_search(elements);
}

#[test]
fn build_and_search_int8() {
    const DIM: usize = 32;

    let elements: Vec<angular_int::Vector> = (0..500)
        .map(|_| test_helper::random_vector::<angular_int::Vector>(DIM))
        .collect();

    build_and_search(elements);
}

#[test]
fn incremental_build_0() {
    let elements: Vec<_> = (0..1000)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    let layer_multiplier = compute_layer_multiplier(elements.len(), config.num_layers);

    let mut builder = GranneBuilder::new(config, elements.as_slice());

    builder.build_index_part(layer_multiplier.ceil() as usize + 2);

    assert_eq!(3, builder.layers.len());
    assert_eq!(
        layer_multiplier.powf(0.).ceil() as usize,
        builder.layers[0].len()
    );
    assert_eq!(
        layer_multiplier.powf(1.).ceil() as usize,
        builder.layers[1].len()
    );
    assert_eq!(
        layer_multiplier.ceil() as usize + 2,
        builder.layers[2].len()
    );

    builder.build_index_part(layer_multiplier.powf(2.).ceil() as usize + 2);

    assert_eq!(4, builder.layers.len());
    assert_eq!(
        layer_multiplier.powf(0.).ceil() as usize,
        builder.layers[0].len()
    );
    assert_eq!(
        layer_multiplier.powf(1.).ceil() as usize,
        builder.layers[1].len()
    );
    assert_eq!(
        layer_multiplier.powf(2.).ceil() as usize,
        builder.layers[2].len()
    );
    assert_eq!(
        layer_multiplier.powf(2.).ceil() as usize + 2,
        builder.layers[3].len()
    );

    builder.build_index_part(elements.len());

    assert_eq!(4, builder.layers.len());
    assert_eq!(
        layer_multiplier.powf(0.).ceil() as usize,
        builder.layers[0].len()
    );
    assert_eq!(
        layer_multiplier.powf(1.).ceil() as usize,
        builder.layers[1].len()
    );
    assert_eq!(
        layer_multiplier.powf(2.).ceil() as usize,
        builder.layers[2].len()
    );
    assert_eq!(elements.len(), builder.layers[3].len());

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn incremental_build_1() {
    let elements: angular_int::Vectors = (0..1000)
        .map(|_| test_helper::random_vector::<angular_int::Vector>(25))
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut builder = GranneBuilder::new(config.clone(), elements.borrow());

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

/*
#[test]
fn incremental_build_with_write_and_read() {
    const DIM: usize = 25;

    let elements: angular::Vectors = (0..1000)
        .map(|_| test_helper::random_vector::<angular::Vector>(DIM))
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    let num_chunks = 10;
    let chunk_size = elements.len() / num_chunks;
    assert_eq!(elements.len(), num_chunks * chunk_size);

    let mut file: File = tempfile::tempfile().unwrap();
    {
        let builder = GranneBuilder::new(config.clone(), elements);

        builder.write_index(&mut file).unwrap();
    }

    for i in 0..num_chunks {
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut builder = GranneBuilder::read(config.clone(), &mut file, elements);
        assert_eq!(i * chunk_size, builder.indexed_elements());

        builder.build_index_part((i + 1) * chunk_size);

        assert_eq!((i + 1) * chunk_size, builder.indexed_elements());

        file.seek(SeekFrom::Start(0)).unwrap();
        builder.write_index(&mut file).unwrap();
    }

    file.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let index = Granne::<angular::Vectors>::load(buffer.as_slice(), &elements);
    assert_eq!(config.num_layers, index.num_layers());
    assert_eq!(elements.len(), index.len());

    verify_search(&index, 0.95, 40);
}

#[test]
fn read_index_with_owned_elements() {
    let num_elements = 1000;
    const DIM: usize = 25;
    type Element = angular::Vector<'static>;

    let mut owning_builder = {
        let elements: Vec<Element> = (0..num_elements)
            .map(|_| test_helper::random_vector::<Element>(DIM))
            .collect();

        let config = Config {
            num_layers: 4,
            num_neighbors: 20,
            max_search: 50,
            reinsert_elements: true,
            show_progress: false,
        };

        let mut builder = GranneBuilder::with_borrowed_elements(config.clone(), elements.as_slice());
        builder.build_index_part(num_elements / 2);

        let mut file: File = tempfile::tempfile().unwrap();
        builder.write(&mut file).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();
        let owning_builder: GranneBuilder<[Element], Element> =
            GranneBuilder::read_index_with_owned_elements(config.clone(), &mut file, elements.clone()).unwrap();

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
*/
#[test]
fn empty_build() {
    let elements: angular::Vectors = (0..100)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut builder = GranneBuilder::new(config, elements);

    builder.build_index_part(0);
}

#[test]
fn test_layer_multiplier() {
    assert_eq!(2, compute_layer_multiplier(10, 5).ceil() as usize);
    assert_eq!(14, compute_layer_multiplier(400000, 6).ceil() as usize);
    assert_eq!(22, compute_layer_multiplier(2000000000, 8).ceil() as usize);
    assert_eq!(555, compute_layer_multiplier(555, 2).ceil() as usize);
    assert_eq!(25, compute_layer_multiplier(625, 3).ceil() as usize);
}

#[test]
fn write_and_load() {
    const DIM: usize = 50;
    let elements: angular::Vectors = (0..100).map(|_| test_helper::random_vector(DIM)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 10,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut builder = GranneBuilder::new(config, elements.borrow());
    builder.build_index();

    let mut file: File = tempfile::tempfile().unwrap();
    builder.write_index(&mut file).unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    assert!(data.len() > 2000);

    let index = Granne::load(&data[..], &elements);

    assert_eq!(builder.layers.len(), index.num_layers());
    assert_eq!(builder.indexed_elements(), index.len());

    for layer in 0..builder.layers.len() {
        for i in 0..builder.layers[layer].len() {
            let builder_neighbors: Vec<_> = builder.layers[layer].get_neighbors(i);

            assert_eq!(builder_neighbors, index.get_neighbors(i, layer));
        }
    }

    assert_eq!(builder.elements.len(), index.len());

    for i in 0..builder.elements.len() {
        assert!(
            builder
                .elements
                .dist_to_element(i, &index.get_element(i))
                .into_inner()
                < DIST_EPSILON
        );
    }
}

#[test]
fn write_and_load_compressed() {
    const DIM: usize = 50;
    let elements: angular::Vectors = (0..100).map(|_| test_helper::random_vector(DIM)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 10,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut builder = GranneBuilder::new(config, elements.borrow());
    builder.build_index();

    {
        let mut file: File = tempfile::tempfile().unwrap();
        builder.write_index(&mut file).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        let index = Granne::load(&data[..], &elements);

        assert_eq!(builder.layers.len(), index.num_layers());
        assert_eq!(builder.indexed_elements(), index.len());

        for layer in 0..builder.layers.len() {
            for i in 0..builder.layers[layer].len() {
                let builder_neighbors = builder.layers[layer].get_neighbors(i);

                assert_eq!(builder_neighbors.len(), index.get_neighbors(i, layer).len());
                assert_eq!(builder_neighbors, index.get_neighbors(i, layer));
            }
        }

        assert_eq!(builder.elements.len(), index.len());

        for i in 0..builder.elements.len() {
            assert!(
                builder
                    .elements
                    .dist_to_element(i, &index.get_element(i))
                    .into_inner()
                    < DIST_EPSILON
            );
        }
        /*
                // write the already compressed index to file and reload
                let mut compressed_file: File = tempfile::tempfile().unwrap();

                io::save_index_to_disk(&index.layers, &mut compressed_file, true).unwrap();

                compressed_file.seek(SeekFrom::Start(0)).unwrap();
                let mut data = Vec::new();
                compressed_file.read_to_end(&mut data).unwrap();

                let index = Granne::<angular::Vectors>::load(&data[..], &elements);

                assert_eq!(builder.layers.len(), index.num_layers());

                for layer in 0..builder.layers.len() {
                    assert_eq!(builder.layers[layer].len(), index.layer_len(layer));

                    for i in 0..builder.layers[layer].len() {
                        let builder_neighbors: Vec<_> = iter_neighbors(builder.layers[layer].get(i)).collect();

                        assert_eq!(builder_neighbors, index.get_neighbors(i, layer));
                    }
                }

                assert_eq!(builder.elements.len(), index.len());

                for i in 0..builder.elements.len() {
                    assert!(builder.elements.at(i).dist(&index.get_element(i)).into_inner() < DIST_EPSILON);
                }
        */
    }
}
/*
#[test]
fn write_and_read() {
    const DIM: usize = 64;

    let elements: angular_int::Vectors = (0..100)
        .map(|_| test_helper::random_vector::<angular::Vector>(DIM).into())
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 10,
        reinsert_elements: true,
        show_progress: false,
    };

    let mut original = GranneBuilder::with_borrowed_elements(config.clone(), &elements);
    original.build_index();

    let mut file: File = tempfile::tempfile().unwrap();
    original.write(&mut file).unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();

    let copy = GranneBuilder::<angular_int::Vectors, angular_int::Vector>::read_index_with_borrowed_elements(
        config, &mut file, &elements,
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
*/
#[test]
fn append_elements() {
    let elements: Vec<angular::Vector> = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(50))
        .collect();

    let additional_elements: Vec<angular::Vector> = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(50))
        .collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    // insert half of the elements
    let mut builder = GranneBuilder::new(config, angular::Vectors::new());
    for element in &elements {
        builder.push(element.clone());
    }
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(500, builder.layers[3].len());

    let max_search = 50;

    // assert that one arbitrary element is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements[123], max_search, 1)
            .iter()
            .any(|&(idx, _)| 123 == idx,));
    }

    // insert rest of the elements
    for element in &additional_elements {
        builder.push(element.clone());
    }
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(1000, builder.layers[3].len());

    // assert that the same arbitrary element and a newly added one
    // is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements[123], max_search, 1)
            .iter()
            .any(|&(idx, _)| 123 == idx));

        assert!(index
            .search(&additional_elements[123], max_search, 1)
            .iter()
            .any(|&(idx, _)| elements.len() + 123 == idx));
    }
}
/*
#[test]
fn append_elements_with_expected_size() {
    let elements: Vec<angular::Vector> = (0..10).map(|_| test_helper::random_vector::<angular::Vector>(50)).collect();

    let additional_elements: Vec<angular::Vector> =
        (10..1000).map(|_| test_helper::random_vector::<angular::Vector>(50)).collect();

    let config = Config {
        num_layers: 4,
        num_neighbors: 20,
        max_search: 50,
        reinsert_elements: true,
        show_progress: false,
    };

    // insert half of the elements
    let mut builder = GranneBuilder::with_expected_size(config, 1000);
    for element in &elements {
        builder.append(element.clone());
    }
    builder.build_index();

    assert_eq!(2, builder.layers.len());
    assert_eq!(10, builder.layers.last().unwrap().len());

    let max_search = 10;

    // assert that one arbitrary element is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements.at(7), 1, max_search)
            .iter()
            .any(|&(idx, _)| 7 == idx,));
    }

    // insert rest of the elements
    for element in &additional_elements {
        builder.append(element.clone());
    }
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(1000, builder.layers[3].len());

    // assert that the same arbitrary element and a newly added one
    // is findable
    {
        let index = builder.get_index();

        assert!(index
            .search(&elements.at(7), 1, max_search)
            .iter()
            .any(|&(idx, _)| 7 == idx,));

        assert!(index
            .search(&additional_elements.at(123), 1, max_search)
            .iter()
            .any(|&(idx, _)| elements.len() + 123 == idx,));
    }
}

*/
