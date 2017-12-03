#![feature(test)]

extern crate test;
extern crate hnsw;

use test::Bencher;

use hnsw::*;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod types {
    use super::*;

    #[bench]
    fn float_reference_dist(b: &mut Bencher) {
        let x = hnsw::example::random_float_element();
        let y = hnsw::example::random_float_element();

        b.iter(|| hnsw::reference_dist(&x, &y));
    }

    #[bench]
    fn float_dist(b: &mut Bencher) {
        let x = hnsw::example::random_float_element();
        let y = hnsw::example::random_float_element();

        b.iter(|| x.dist(&y));
    }

    #[bench]
    fn normalized_float_dist(b: &mut Bencher) {
        let x = hnsw::example::random_float_element().normalized();
        let y = hnsw::example::random_float_element().normalized();

        b.iter(|| x.dist(&y));
    }

    #[bench]
    fn int8_dist(b: &mut Bencher) {
        let x = hnsw::example::random_int8_element();
        let y = hnsw::example::random_int8_element();

        b.iter(|| x.dist(&y));
    }
}
