#![feature(test)]

extern crate granne;
extern crate test;

use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

mod query_embeddings {
    use super::*;
    use granne::query_embeddings::*;

    #[bench]
    fn get_8_words(b: &mut Bencher) {
        let mut qv = QueryVec::new();
        let ids: Vec<usize> = (0..8).collect();
        qv.push(&ids[..]);

        b.iter(|| qv.get(0));
    }

    #[bench]
    fn get_4_words(b: &mut Bencher) {
        let mut qv = QueryVec::new();
        let ids: Vec<usize> = (0..4).collect();
        qv.push(&ids[..]);

        b.iter(|| qv.get(0));
    }

    #[bench]
    fn get_1_word(b: &mut Bencher) {
        let mut qv = QueryVec::new();
        let ids: Vec<usize> = (0..1).collect();
        qv.push(&ids[..]);

        b.iter(|| qv.get(0));
    }

    #[inline(never)]
    fn reference_words_impl(input: &[u32; 8]) -> [u32; 8] {
        input.clone()
    }

    #[bench]
    fn get_words_reference(b: &mut Bencher) {
        let ids = [13u32; 8];

        b.iter(|| reference_words_impl(&ids));
    }

    #[bench]
    fn into_element_2_words(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(100, 1000);

        let query = vec![73, 19];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }

    #[bench]
    fn into_element_4_words(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(100, 1000);

        let query = vec![513, 37, 566, 2];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }

    #[bench]
    fn into_element_8_words(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(100, 1000);

        let query = vec![0, 150, 255, 77, 12, 3, 55, 599];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }

    #[bench]
    fn into_element_2_words_300(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(300, 1000);

        let query = vec![73, 19];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }

    #[bench]
    fn into_element_4_words_300(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(300, 1000);

        let query = vec![513, 37, 566, 2];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }

    #[bench]
    fn into_element_8_words_300(b: &mut Bencher) {
        let word_embeddings = get_random_word_embeddings(300, 1000);

        let query = vec![0, 150, 255, 77, 12, 3, 55, 599];

        b.iter(|| word_embeddings.get_embedding(&query[..]));
    }
}
