#![feature(test)]

extern crate test;

use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod index {
    use super::*;
    use granne::*;
    use std::io::{Read, Seek, SeekFrom};
    use tempfile;

    macro_rules! builder_search_impl {
        ($fn_name:ident, $elements_type:ty, $dim:expr, $max_search:expr) => {
            #[bench]
            fn $fn_name(b: &mut Bencher) {
                let elements: $elements_type = (0..1000).map(|_| test_helper::random_vector($dim)).collect();

                let mut builder = GranneBuilder::new(
                    BuildConfig::default()
                        .layer_multiplier(15.0)
                        .num_neighbors(20)
                        .max_search(50),
                    elements,
                );

                builder.build();

                let index = builder.get_index();

                b.iter(|| {
                    for i in (0..index.len()).step_by(10) {
                        index.search(&index.get_element(i), $max_search, 1);
                    }
                });
            }
        };
    }

    builder_search_impl!(builder_search_angular_vector_3_ms_50, angular::Vectors, 3, 50);
    builder_search_impl!(builder_search_angular_vector_100_ms_50, angular::Vectors, 100, 50);

    builder_search_impl!(builder_search_angular_int_vector_3_ms_50, angular_int::Vectors, 3, 50);
    builder_search_impl!(
        builder_search_angular_int_vector_100_ms_50,
        angular_int::Vectors,
        100,
        50
    );

    macro_rules! index_search_impl {
        ($fn_name:ident, $elements_type:ty, $dim:expr, $max_search:expr) => {
            #[bench]
            fn $fn_name(b: &mut Bencher) {
                let elements: $elements_type = (0..1000).map(|_| test_helper::random_vector($dim)).collect();

                let mut builder = GranneBuilder::new(
                    BuildConfig::default()
                        .layer_multiplier(15.0)
                        .num_neighbors(20)
                        .max_search(50),
                    elements,
                );

                builder.build();

                let mut file: std::fs::File = tempfile::tempfile().unwrap();
                builder.write_index(&mut file).unwrap();
                file.seek(SeekFrom::Start(0)).unwrap();
                let mut data = Vec::new();
                file.read_to_end(&mut data).unwrap();

                let index = Granne::from_bytes(&data[..], builder.get_elements());

                b.iter(|| {
                    for i in (0..index.len()).step_by(10) {
                        index.search(&index.get_element(i), $max_search, 1);
                    }
                });
            }
        };
    }

    index_search_impl!(index_search_angular_vector_3_ms_50, angular::Vectors, 3, 50);
    index_search_impl!(index_search_angular_vector_100_ms_50, angular::Vectors, 100, 50);

    index_search_impl!(index_search_angular_int_vector_3_ms_50, angular_int::Vectors, 3, 50);
    index_search_impl!(index_search_angular_int_vector_100_ms_50, angular_int::Vectors, 100, 50);
}
