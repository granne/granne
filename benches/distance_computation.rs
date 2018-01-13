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
    fn float_reference_dist(b: &mut Bencher) {
        let x = granne::example::random_float_element();
        let y = granne::example::random_float_element();

        b.iter(|| granne::reference_dist(&x, &y));
    }

    #[bench]
    fn float_dist(b: &mut Bencher) {
        let x = granne::example::random_float_element();
        let y = granne::example::random_float_element();

        b.iter(|| x.dist(&y));
    }

    #[bench]
    fn normalized_float_dist(b: &mut Bencher) {
        let x = granne::example::random_float_element().normalized();
        let y = granne::example::random_float_element().normalized();

        b.iter(|| x.dist(&y));
    }

    #[bench]
    fn int8_dist(b: &mut Bencher) {
        let x = granne::example::random_int8_element();
        let y = granne::example::random_int8_element();

        b.iter(|| x.dist(&y));
    }
}
