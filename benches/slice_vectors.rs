#![feature(test)]

extern crate test;

use granne::*;
use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}
mod slice_vectors {
    use super::*;

    #[bench]
    fn offsets_get_10(b: &mut Bencher) {
        let mut offsets = Offsets::new();

        for offset in test_helper::random_offsets(u16::max_value() as usize).take(10_000) {
            offsets.push(offset);
        }

        let mut sum = 0;
        b.iter(|| {
            sum += offsets.get(9999);
            sum += offsets.get(5123);
            sum += offsets.get(599);
            sum += offsets.get(435);
            sum += offsets.get(0);
            sum += offsets.get(1800);
            sum += offsets.get(888);
            sum += offsets.get(1221);
            sum += offsets.get(64);
            sum += offsets.get(7200);
        });
    }

    #[bench]
    fn vec_get_10(b: &mut Bencher) {
        let mut offsets = Vec::new();

        for offset in test_helper::random_offsets(u16::max_value() as usize).take(10_000) {
            offsets.push(offset);
        }

        let mut sum = 0;
        b.iter(|| {
            sum += offsets.get(9999).unwrap();
            sum += offsets.get(5123).unwrap();
            sum += offsets.get(599).unwrap();
            sum += offsets.get(435).unwrap();
            sum += offsets.get(0).unwrap();
            sum += offsets.get(1800).unwrap();
            sum += offsets.get(888).unwrap();
            sum += offsets.get(1221).unwrap();
            sum += offsets.get(64).unwrap();
            sum += offsets.get(7200).unwrap();
        });
    }

    #[bench]
    fn var_width_get_10(b: &mut Bencher) {
        let mut vec = VariableWidthSliceVector::<_, usize>::new();

        for slice in (0..10_000).map(|i| (i..).take((1 + i) % 40).collect::<Vec<_>>()) {
            vec.push(&slice);
        }

        let mut sum = 0;
        b.iter(|| {
            sum += vec.get(9999).len();
            sum += vec.get(5123).len();
            sum += vec.get(599).len();
            sum += vec.get(435).len();
            sum += vec.get(0).len();
            sum += vec.get(1800).len();
            sum += vec.get(888).len();
            sum += vec.get(1221).len();
            sum += vec.get(64).len();
            sum += vec.get(7200).len();
        });
    }

    #[bench]
    fn set_vec_get_10(b: &mut Bencher) {
        let mut vec = MultiSetVector::new();

        for slice in
            (0..10_000).map(|i: u32| (i..).take((1 + i as usize) % 40).collect::<Vec<u32>>())
        {
            vec.push(&slice);
        }

        let mut sum = 0;
        let mut res = Vec::new();
        b.iter(|| {
            vec.get_into(9999, &mut res);
            vec.get_into(5123, &mut res);
            vec.get_into(599, &mut res);
            vec.get_into(435, &mut res);
            vec.get_into(0, &mut res);
            vec.get_into(1800, &mut res);
            vec.get_into(888, &mut res);
            vec.get_into(1221, &mut res);
            vec.get_into(64, &mut res);
            vec.get_into(7200, &mut res);
        });
    }

    #[bench]
    fn compressed_var_width_get_10(b: &mut Bencher) {
        let mut vec = CompressedVariableWidthSliceVector::new();

        for slice in (0..10_000).map(|i| (i..).take((1 + i) % 40).collect::<Vec<_>>()) {
            vec.push(&slice);
        }

        let mut sum = 0;
        b.iter(|| {
            sum += vec.get(9999).len();
            sum += vec.get(5123).len();
            sum += vec.get(599).len();
            sum += vec.get(435).len();
            sum += vec.get(0).len();
            sum += vec.get(1800).len();
            sum += vec.get(888).len();
            sum += vec.get(1221).len();
            sum += vec.get(64).len();
            sum += vec.get(7200).len();
        });
    }
}
