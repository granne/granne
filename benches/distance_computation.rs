#![feature(test)]

extern crate test;

use test::Bencher;

use granne::*;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod types {
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

    vector_dist_impl!(angular_vector_003_dist, AngularVector, 3);
    vector_dist_impl!(angular_vector_050_dist, AngularVector, 50);
    vector_dist_impl!(angular_vector_100_dist, AngularVector, 100);
    vector_dist_impl!(angular_vector_200_dist, AngularVector, 200);
    vector_dist_impl!(angular_vector_300_dist, AngularVector, 300);

    vector_dist_impl!(angular_int_vector_003_dist, AngularIntVector, 3);
    vector_dist_impl!(angular_int_vector_050_dist, AngularIntVector, 50);
    vector_dist_impl!(angular_int_vector_100_dist, AngularIntVector, 100);
    vector_dist_impl!(angular_int_vector_200_dist, AngularIntVector, 200);
    vector_dist_impl!(angular_int_vector_300_dist, AngularIntVector, 300);
}
