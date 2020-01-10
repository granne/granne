#![feature(test)]

extern crate test;

use test::Bencher;

use granne::*;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod elements {
    use super::*;

    macro_rules! vector_dist_impl {
        ($fn_name:ident, $vector_type:ty, $dim:expr) => {
            #[bench]
            fn $fn_name(b: &mut Bencher) {
                let x: $vector_type = granne::test_helper::random_vector($dim);
                let y: $vector_type = granne::test_helper::random_vector($dim);

                b.iter(|| x.dist(&y));
            }
        };
    }

    vector_dist_impl!(angular_vector_003_dist, angular::Vector, 3);
    vector_dist_impl!(angular_vector_050_dist, angular::Vector, 50);
    vector_dist_impl!(angular_vector_100_dist, angular::Vector, 100);
    vector_dist_impl!(angular_vector_200_dist, angular::Vector, 200);
    vector_dist_impl!(angular_vector_300_dist, angular::Vector, 300);

    vector_dist_impl!(angular_int_vector_003_dist, angular_int::Vector, 3);
    vector_dist_impl!(angular_int_vector_050_dist, angular_int::Vector, 50);
    vector_dist_impl!(angular_int_vector_100_dist, angular_int::Vector, 100);
    vector_dist_impl!(angular_int_vector_200_dist, angular_int::Vector, 200);
    vector_dist_impl!(angular_int_vector_300_dist, angular_int::Vector, 300);
}
