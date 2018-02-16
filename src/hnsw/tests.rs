use super::*;
use std::mem;
use types::example::*;
use types::*;
use file_io;

#[test]
fn hnsw_node_size() {
    assert!((MAX_NEIGHBORS) * mem::size_of::<NeighborType>() <= mem::size_of::<HnswNode>());

    assert!(
        mem::size_of::<HnswNode>() <=
            MAX_NEIGHBORS * mem::size_of::<NeighborType>() + mem::size_of::<usize>()
    );
}

#[test]
fn select_neighbors() {
    let element: AngularVector<[f32; 50]> = random_dense_element();

    let other_elements: Vec<AngularVector<[f32; 50]>> = (0..50).map(|_| random_dense_element()).collect();

    let candidates: Vec<_> = other_elements
        .iter()
        .map(|e| e.dist(&element))
        .enumerate()
        .collect();

    let neighbors = HnswBuilder::select_neighbors(&other_elements[..], candidates.clone(), 10);

    assert_eq!(10, neighbors.len());


    let neighbors = HnswBuilder::select_neighbors(&other_elements[..], candidates.clone(), 60);

    assert_eq!(50, neighbors.len());
}

fn build_and_search<T: ComparableTo<T> + Sync + Send + Clone>(elements: Vec<T>) {
    let config = Config {
        num_layers: 5,
        max_search: 50,
        show_progress: false,
    };

    let mut builder = HnswBuilder::new(config);
    builder.add(elements.clone());
    builder.build_index();
    let index = builder.get_index();

    verify_search(&index, 0.95);
}

fn verify_search<T: ComparableTo<T> + Sync + Send + Clone>(index: &Hnsw<T,T>, precision: f32) {
    let max_search = 40;
    let mut num_found = 0;
    for (i, element) in index.elements.iter().enumerate() {
        if index.search(element, 1, max_search)[0].0 == i {
            num_found += 1;
        }
    }

    let p1 = num_found as f32 / index.len() as f32;

    println!("p1: {}", p1);
    assert!(precision < p1);
}

#[test]
fn with_elements() {
    let config = Config {
        num_layers: 5,
        max_search: 50,
        show_progress: false
    };

    let elements: Vec<_> = (0..500).map(|_| random_dense_element::<AngularVector<[f32; 25]>, _>()).collect();

    let mut builder = HnswBuilder::with_elements(config, &elements);

    assert_eq!(elements.len(), builder.elements.len());
    builder.build_index();
    let index = builder.get_index();

    assert_eq!(index.len(), elements.len());

    verify_search(&index, 0.95);
}

#[test]
fn with_elements_and_add() {
    let config = Config {
        num_layers: 5,
        max_search: 50,
        show_progress: false
    };

    let elements: Vec<_> = (0..600).map(|_| random_dense_element::<AngularVector<[f32; 25]>, _>()).collect();

    let mut builder = HnswBuilder::with_elements(config, &elements[..500]);

    assert_eq!(500, builder.elements.len());

    builder.add(elements[500..].to_vec());

    assert_eq!(600, builder.elements.len());

    builder.build_index();

    verify_search(&builder.get_index(), 0.95);
}

#[test]
fn build_and_search_float() {
    let elements: Vec<_> = (0..1500).map(|_| random_dense_element::<AngularVector<[f32; 128]>, _>()).collect();

    build_and_search(elements);
}

#[test]
fn build_and_search_int8() {
    let elements: Vec<AngularIntVector<[i8; 32]>> =
        (0..500)
        .map(|_| random_dense_element::<AngularVector<[f32; 32]>, _>().into())
        .collect();

    build_and_search(elements);
}

#[test]
fn test_layer_multiplier() {
    assert_eq!(
        2,
        compute_layer_multiplier(10, 5)
    );
    assert_eq!(
        14,
        compute_layer_multiplier(400000, 6)
    );
    assert_eq!(
        22,
        compute_layer_multiplier(2000000000, 8)
    );
    assert_eq!(
        555,
        compute_layer_multiplier(555, 2)
    );
    assert_eq!(
        25,
        compute_layer_multiplier(625, 3)
    );
}

#[test]
fn write_and_load() {
    let elements: Vec<AngularVector<[f32; 50]>> = (0..100).map(|_| random_dense_element()).collect();

    let config = Config {
        num_layers: 4,
        max_search: 10,
        show_progress: false,
    };

    let mut builder = HnswBuilder::new(config);
    builder.add(elements.clone());
    builder.build_index();

    let mut data = Vec::new();
    builder.write(&mut data).unwrap();

    let index = Hnsw::<AngularVector<[f32; 50]>, AngularVector<[f32; 50]>>::load(&data[..], &elements[..]);

    assert_eq!(builder.layers.len(), index.layers.len());

    for layer in 0..builder.layers.len() {
        assert_eq!(builder.layers[layer].len(), index.layers[layer].len());

        for i in 0..builder.layers[layer].len() {
            assert_eq!(
                builder.layers[layer][i].neighbors,
                index.layers[layer][i].neighbors
            );
        }
    }

    assert_eq!(builder.elements.len(), index.elements.len());

    for i in 0..builder.elements.len() {
        assert!(builder.elements[i].dist(&index.elements[i]).into_inner() < DIST_EPSILON);
    }
}

#[test]
fn write_and_read() {
    const DIM: usize = 64;

    let elements: Vec<AngularIntVector<[i8; DIM]>> =
        (0..100)
        .map(|_| random_dense_element::<AngularVector<[f32; DIM]>, _>().into())
        .collect();

    let config = Config {
        num_layers: 4,
        max_search: 10,
        show_progress: false,
    };

    let mut original = HnswBuilder::new(config);
    original.add(elements.clone());
    original.build_index();

    let mut data = Vec::new();
    original.write(&mut data).unwrap();

    let mut elements_data = Vec::new();
    file_io::write(&elements, &mut elements_data).unwrap();

    let copy = HnswBuilder::<AngularIntVector<[i8; DIM]>>::read(&mut data.as_slice(), &mut elements_data.as_slice()).unwrap();

    assert_eq!(original.layers.len(), copy.layers.len());

    for layer in 0..original.layers.len() {
        assert_eq!(original.layers[layer].len(), copy.layers[layer].len());

        for i in 0..original.layers[layer].len() {
            assert_eq!(
                original.layers[layer][i].neighbors,
                copy.layers[layer][i].neighbors
            );
        }
    }

    assert_eq!(original.elements.len(), copy.elements.len());

    for i in 0..original.elements.len() {
        assert!(
            original.elements[i].0.as_slice().iter().zip(
                copy.elements[i].0.as_slice().iter()).all(|(x,y)| x == y),
            "Elements with index {} differ",
            i
        );
    }
}

#[test]
fn append_elements() {
    let elements: Vec<_> = (0..1000)
        .map(|_| random_dense_element::<AngularVector<[f32; 50]>, _>())
        .collect();

    let config = Config {
        num_layers: 4,
        max_search: 50,
        show_progress: false,
    };

    // insert half of the elements
    let mut builder = HnswBuilder::new(config);
    builder.add(elements[..500].to_vec());
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(500, builder.layers[3].len());

    let max_search = 50;

    // assert that one arbitrary element is findable
    {
        let index = builder.get_index();

        assert!(index.search(&elements[123], 1, max_search).iter().any(
            |&(idx, _)| 123 == idx,
        ));
    }

    // insert rest of the elements
    builder.add(elements[500..].to_vec());
    builder.build_index();

    assert_eq!(4, builder.layers.len());
    assert_eq!(1000, builder.layers[3].len());

    // assert that the same arbitrary element and a newly added one
    // is findable
    {
        let index = builder.get_index();

        assert!(index.search(&elements[123], 1, max_search).iter().any(
            |&(idx, _)| 123 == idx,
        ));

        assert!(index.search(&elements[789], 1, max_search).iter().any(
            |&(idx, _)| 789 == idx,
        ));
    }
}
