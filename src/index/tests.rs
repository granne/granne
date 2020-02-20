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

    let other_elements: Vec<angular::Vector> = (0..50).map(|_| test_helper::random_vector(DIM)).collect();

    let mut candidates: Vec<_> = other_elements.iter().map(|e| e.dist(&element)).enumerate().collect();

    candidates.sort_unstable_by_key(|&(_, d)| d);

    let neighbors = GranneBuilder::select_neighbors(&other_elements.as_slice(), candidates.clone(), 10);

    assert!(0 < neighbors.len() && neighbors.len() <= 10);

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }

    let neighbors = GranneBuilder::select_neighbors(&other_elements.as_slice(), candidates.clone(), 60);

    assert_eq!(candidates.len(), neighbors.len());

    // assert that neighbors are sorted on distance from element
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 <= neighbors[i].1);
    }
}

fn build_and_search<Elements: ElementContainer + Sync + Send + Clone>(elements: Elements) {
    let mut builder = GranneBuilder::new(BuildConfig::default().num_neighbors(20).max_search(20), elements);

    builder.build();
    let index = builder.get_index();

    verify_search(&index, 0.95, 10);
}

fn verify_search<Elements: ElementContainer>(index: &Granne<Elements>, precision: f32, max_search: usize) {
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
    let elements: Vec<_> = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let mut builder = GranneBuilder::new(
        BuildConfig::default().max_search(5).reinsert_elements(false),
        elements.as_slice(),
    );

    assert_eq!(elements.len(), builder.elements.len());
    builder.build();
    let index = builder.get_index();

    assert_eq!(index.len(), elements.len());

    verify_search(&index, 0.95, 40);
}

#[test]
fn with_elements_and_add() {
    let elements: angular::Vectors = (0..500)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();
    let additional_elements: Vec<angular::Vector> = (0..100)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let mut builder = GranneBuilder::new(
        BuildConfig::default()
            .expected_num_elements(600)
            .num_neighbors(20)
            .max_search(5),
        elements,
    );

    assert_eq!(500, builder.elements.len());

    for element in additional_elements {
        builder.push(element);
    }

    assert_eq!(600, builder.elements.len());

    builder.build();

    verify_search(&builder.get_index(), 0.95, 40);
}

#[test]
fn build_and_search_float() {
    let elements: Vec<_> = (0..1500)
        .map(|_| test_helper::random_vector::<angular::Vector>(28))
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
        .map(|_| test_helper::random_vector::<angular::Vector>(5))
        .collect();

    let mut builder = GranneBuilder::new(
        BuildConfig::default()
            .layer_multiplier(10.0)
            .num_neighbors(20)
            .max_search(5),
        elements.as_slice(),
    );

    builder.build_partial(12);

    assert_eq!(2, builder.layers.len());
    assert_eq!(10, builder.layers[0].len());
    assert_eq!(12, builder.layers[1].len());

    builder.build_partial(102);

    assert_eq!(3, builder.layers.len());
    assert_eq!(10, builder.layers[0].len());
    assert_eq!(100, builder.layers[1].len());
    assert_eq!(102, builder.layers[2].len());

    builder.build();

    assert_eq!(3, builder.layers.len());
    assert_eq!(10, builder.layers[0].len());
    assert_eq!(100, builder.layers[1].len());
    assert_eq!(1000, builder.layers[2].len());

    verify_search(&builder.get_index(), 0.95, 5);
}

#[test]
fn incremental_build_1() {
    let elements: angular_int::Vectors = (0..1000)
        .map(|_| test_helper::random_vector::<angular_int::Vector>(5))
        .collect();

    let mut builder = GranneBuilder::new(BuildConfig::default().max_search(50), elements.borrow());

    let num_chunks = 10;
    let chunk_size = elements.len() / num_chunks;
    assert_eq!(elements.len(), num_chunks * chunk_size);

    for i in 1..(num_chunks + 1) {
        builder.build_partial(i * chunk_size);

        assert_eq!(i * chunk_size, builder.len());
    }

    assert_eq!(elements.len(), builder.len());

    verify_search(&builder.get_index(), 0.95, 5);
}

#[test]
fn incremental_build_with_write_and_read() {
    const DIM: usize = 25;

    let elements: angular::Vectors = (0..1000)
        .map(|_| test_helper::random_vector::<angular::Vector>(DIM))
        .collect();

    let config = BuildConfig::default().reinsert_elements(false);

    let num_chunks = 10;
    let chunk_size = elements.len() / num_chunks;
    assert_eq!(elements.len(), num_chunks * chunk_size);

    let mut file: File = tempfile::tempfile().unwrap();
    {
        let builder = GranneBuilder::new(config.clone(), &elements);

        builder.write_index(&mut file).unwrap();
    }

    for i in 0..num_chunks {
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut builder = {
            let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };

            GranneBuilder::from_bytes(config.clone(), &mmap[..], &elements)
        };

        assert_eq!(i * chunk_size, builder.len());

        builder.build_partial((i + 1) * chunk_size);

        assert_eq!((i + 1) * chunk_size, builder.len());

        file.seek(SeekFrom::Start(0)).unwrap();
        builder.write_index(&mut file).unwrap();
    }

    file.seek(SeekFrom::Start(0)).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let index = Granne::from_bytes(buffer.as_slice(), &elements);
    assert_eq!(elements.len(), index.len());

    verify_search(&index, 0.95, 40);
}

#[test]
fn read_index_reduce_num_neighbors() {
    let num_elements = 1000;
    const DIM: usize = 5;
    type Element = angular::Vector<'static>;

    let mut owning_builder = {
        let elements: Vec<Element> = (0..num_elements)
            .map(|_| test_helper::random_vector::<Element>(DIM))
            .collect();

        let config = BuildConfig::default().max_search(10).num_neighbors(20);

        let mut builder = GranneBuilder::new(config, elements.as_slice());
        builder.build_partial(num_elements / 2);

        let mut file: File = tempfile::tempfile().unwrap();
        builder.write_index(&mut file).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        // not necessarily true, but should be valid
        assert!(builder.get_neighbors(0, builder.layers.len() - 1).len() > 5);

        let owning_builder = {
            let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };

            // lower num_neighbors
            let config = BuildConfig::default().max_search(10).num_neighbors(5);
            GranneBuilder::from_bytes(config, &mmap[..], elements.clone())
        };

        assert_eq!(num_elements / 2, builder.len());
        assert_eq!(num_elements / 2, owning_builder.len());

        assert_eq!(builder.layers.len(), owning_builder.layers.len());

        owning_builder
    };

    // owning_builder still alive after elements go out of scope
    assert_eq!(num_elements / 2, owning_builder.len());

    owning_builder.build();

    assert_eq!(num_elements, owning_builder.len());
    assert!(owning_builder.get_neighbors(0, owning_builder.layers.len() - 1).len() <= 5);
}

#[test]
fn empty_build() {
    let elements: angular::Vectors = (0..100)
        .map(|_| test_helper::random_vector::<angular::Vector>(25))
        .collect();

    let mut builder = GranneBuilder::new(BuildConfig::default(), elements);

    builder.build_partial(0);
}

#[test]
fn test_num_elements_in_layer() {
    fn verify(num_elements: usize, multiplier: f32, expected: &[usize]) {
        let actual: Vec<usize> = (0..expected.len())
            .map(|layer| compute_num_elements_in_layer(num_elements, multiplier, layer))
            .collect();

        assert_eq!(expected, actual.as_slice());
    }

    verify(1000, 10.0, &[10, 100, 1000]);
    verify(32, 2.0, &[1, 2, 4, 8, 16, 32]);
    verify(10_000, 10.0, &[1, 10, 100, 1000, 10_000, 10_000]);
    verify(20, 1.9, &[2, 3, 6, 11, 20, 20]);
    verify(
        1_000_000_000,
        20.0,
        &[
            16,
            313,
            6250,
            125_000,
            2_500_000,
            50_000_000,
            1_000_000_000,
            1_000_000_000,
        ],
    );
    verify(50, 100.0, &[50]);

    verify(133689866, 15.0, &[12, 177, 2641, 39612, 594178, 8912658, 133689866]);
}

#[test]
fn write_and_load() {
    const DIM: usize = 50;
    let elements: angular::Vectors = (0..100).map(|_| test_helper::random_vector(DIM)).collect();

    let mut builder = GranneBuilder::new(
        BuildConfig::default().num_neighbors(20).max_search(5),
        elements.borrow(),
    );

    builder.build();

    let mut file: File = tempfile::tempfile().unwrap();
    builder.write_index(&mut file).unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();

    let index = unsafe { Granne::from_file(&file, &elements).unwrap() };

    assert_eq!(builder.layers.len(), index.num_layers());
    assert_eq!(builder.len(), index.len());

    for layer in 0..builder.layers.len() {
        for i in 0..builder.layers[layer].len() {
            let mut builder_neighbors = builder.layers[layer].get_neighbors(i);
            let mut index_neighbors = index.get_neighbors(i, layer);

            builder_neighbors.sort();
            index_neighbors.sort();

            assert_eq!(builder_neighbors, index_neighbors);
        }
    }

    assert_eq!(builder.elements.len(), index.len());

    for i in 0..builder.elements.len() {
        assert!(builder.elements.dist_to_element(i, &index.get_element(i)).into_inner() < DIST_EPSILON);
    }
}

#[test]
fn write_and_load_compressed() {
    const DIM: usize = 50;
    let elements: angular::Vectors = (0..100).map(|_| test_helper::random_vector(DIM)).collect();

    let mut builder = GranneBuilder::new(
        BuildConfig::default().num_neighbors(20).max_search(10),
        elements.borrow(),
    );

    builder.build();

    {
        let mut file: File = tempfile::tempfile().unwrap();
        builder.write_index(&mut file).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        let index = Granne::from_bytes(&data[..], &elements);

        assert_eq!(builder.layers.len(), index.num_layers());
        assert_eq!(builder.len(), index.len());

        for layer in 0..builder.layers.len() {
            for i in 0..builder.layers[layer].len() {
                let mut builder_neighbors = builder.layers[layer].get_neighbors(i);
                let mut index_neighbors = index.get_neighbors(i, layer);

                builder_neighbors.sort();
                index_neighbors.sort();

                assert_eq!(builder_neighbors, index_neighbors);
            }
        }

        assert_eq!(builder.elements.len(), index.len());

        for i in 0..builder.elements.len() {
            assert!(builder.elements.dist_to_element(i, &index.get_element(i)).into_inner() < DIST_EPSILON);
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
    original.build();

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

    let mut builder = GranneBuilder::new(
        BuildConfig::default()
            .expected_num_elements(1000)
            .layer_multiplier(10.0)
            .num_neighbors(20)
            .max_search(50),
        angular::Vectors::new(),
    );

    // insert half of the elements
    for element in &elements {
        builder.push(element.clone());
    }
    builder.build();

    assert_eq!(3, builder.layers.len());
    assert_eq!(500, builder.layers[2].len());

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
    builder.build();

    assert_eq!(3, builder.layers.len());
    assert_eq!(1000, builder.layers[2].len());

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
    builder.build();

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
    builder.build();

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
