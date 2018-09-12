#![feature(test)]

extern crate test;
extern crate granne;

use test::Bencher;

use granne::*;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod types {
    use super::*;

    #[bench]
    fn angular_100_reference_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularVector =
                granne::example::random_dense_element(100);
            let y: AngularVector
                = granne::example::random_dense_element(100);

            b.iter(|| granne::angular_reference_dist(&x, &y));
        }
    }

    #[bench]
    fn angular_int_vector_100_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularIntVector<[i8; 100]> =
                granne::example::random_dense_element::<AngularVector>(100).into();
            let y: AngularIntVector<[i8; 100]> =
                granne::example::random_dense_element::<AngularVector>(100).into();

            b.iter(|| x.dist(&y));
        }
    }

    #[bench]
    fn angular_vector_300_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularVector =
                granne::example::random_dense_element(300);
            let y: AngularVector =
                granne::example::random_dense_element(300);

            b.iter(|| x.dist(&y));
        }
    }

    #[bench]
    fn angular_vector_100_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularVector =
                granne::example::random_dense_element(100);
            let y: AngularVector =
                granne::example::random_dense_element(100);

            b.iter(|| x.dist(&y));
        }
    }

    #[bench]
    fn angular_vector_50_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularVector =
                granne::example::random_dense_element(50);
            let y: AngularVector =
                granne::example::random_dense_element(50);

            b.iter(|| x.dist(&y));
        }
    }

    #[bench]
    fn angular_vector_3_dist(b: &mut Bencher) {
        for _ in 0..5 {
            let x: AngularVector =
                granne::example::random_dense_element(3);
            let y: AngularVector =
                granne::example::random_dense_element(3);

            b.iter(|| x.dist(&y));
        }
    }
}
