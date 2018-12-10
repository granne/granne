use crate::file_io;
use crate::hnsw::{At, Writeable};
use crate::types::AngularVector;
use crate::types::Dense;
use crate::types;

use blas;
use bytes::{ByteOrder, LittleEndian};
use rand::Rng;
use rand;
use slice_vector::{SliceVector, VariableWidthSliceVector};
use std::convert::Into;
use std::io::{Result, Write};

pub mod parsing;

#[derive(Clone)]
pub struct QueryEmbeddings<'a> {
    word_embeddings: WordEmbeddings<'a>,
    queries: QueryVec<'a>
}

impl<'a> QueryEmbeddings<'a> {
    pub fn new(word_embeddings: WordEmbeddings<'a>) -> Self
    {
        Self {
            word_embeddings: word_embeddings,
            queries: QueryVec::new(),
        }
    }

    pub fn from(word_embeddings: WordEmbeddings<'a>, queries: QueryVec<'a>) -> Self
    {
        Self {
            word_embeddings: word_embeddings,
            queries: queries,
        }
    }

    pub fn load(dimension:usize,
                word_embeddings: &'a [u8],
                query_buffer: &'a [u8]) -> Self
    {
        let word_embeddings = WordEmbeddings::load(dimension, word_embeddings);
        let queries = QueryVec::load(query_buffer);

        Self::from(word_embeddings, queries)
    }

    pub fn get_words(self: &Self, idx: usize) -> Vec<usize>
    {
        self.queries.get(idx).iter().map(|&x| x.into()).collect()
    }

    pub fn get_embedding(self: &Self, idx: usize) -> types::AngularVector<'static>
    {
        self.word_embeddings.get_embedding_internal(self.queries.queries.get(idx))
    }

    pub fn get_embedding_for_query(self: &Self, word_ids: &[usize]) -> types::AngularVector<'static>
    {
        self.word_embeddings.get_embedding(word_ids)
    }

    pub fn len(self: &Self) -> usize {
        self.queries.len()
    }
}

impl<'a> At for QueryEmbeddings<'a> {
    type Output = types::AngularVector<'static>;

    fn at(self: &Self, index: usize) -> Self::Output {
        self.get_embedding(index)
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a> Writeable for QueryEmbeddings<'a> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        self.queries.write(buffer)
    }
}

#[derive(Clone)]
pub struct WordEmbeddings<'a> {
    embeddings: types::AngularVectors<'a>
}

impl<'a> WordEmbeddings<'a> {

    pub fn load(dimension: usize, data: &'a [u8]) -> Self {
        assert_eq!(0, data.len() % (dimension * ::std::mem::size_of::<f32>()));

        let embeddings = types::AngularVectors::load(dimension, data);

        Self { embeddings }
    }

    pub fn get_embedding(self: &Self, word_ids: &[usize]) -> types::AngularVector<'static> {
        let word_ids: Vec<WordId> = word_ids.iter().map(|&w| w.into()).collect();
        self.get_embedding_internal(&word_ids)
    }

    fn get_embedding_internal(self: &Self, word_ids: &[WordId]) -> types::AngularVector<'static>
    {
        if word_ids.is_empty() {
            return vec![0.0f32; self.embeddings.dim].into()
        }

        let w: usize = word_ids[0].into();
        let mut data: Vec<f32> = self.embeddings.at(w).into();

        for &w in word_ids.iter().skip(1) {
            let w: usize = w.into();
            let embedding = self.embeddings.get_element(w);

            unsafe { blas::saxpy(embedding.dim() as i32, 1f32, embedding.as_slice(), 1, data.as_mut_slice(), 1) };
        }

        data.into()
    }

    pub fn len(self: &Self) -> usize {
        self.embeddings.len()
    }
}

const BYTES_PER_OFFSET: usize = 5;
#[repr(C,packed)]
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct Offset([u8; BYTES_PER_OFFSET]);

impl From<usize> for Offset {
    #[inline(always)]
    fn from(integer: usize) -> Self {
        let mut data: [u8; BYTES_PER_OFFSET] = unsafe { ::std::mem::uninitialized() };
        LittleEndian::write_uint(&mut data, integer as u64, BYTES_PER_OFFSET);
        Offset(data)
    }
}

impl Into<usize> for Offset {
    #[inline(always)]
    fn into(self: Self) -> usize {
        LittleEndian::read_uint(&self.0, BYTES_PER_OFFSET) as usize
    }
}

const BYTES_PER_WORD_ID: usize = 3;

#[repr(C,packed)]
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct WordId([u8; BYTES_PER_WORD_ID]);

impl From<usize> for WordId {
    #[inline(always)]
    fn from(integer: usize) -> Self {
        let mut data: [u8; BYTES_PER_WORD_ID] = unsafe { ::std::mem::uninitialized() };
        LittleEndian::write_uint(&mut data, integer as u64, BYTES_PER_WORD_ID);
        WordId(data)
    }
}

impl Into<usize> for WordId {
    #[inline(always)]
    fn into(self: Self) -> usize {
        LittleEndian::read_uint(&self.0, BYTES_PER_WORD_ID) as usize
    }
}


/// A Vec for storing variable lengths queries (= sequence of word ids)
#[derive(Clone)]
pub struct QueryVec<'a> {
    queries: VariableWidthSliceVector<'a, WordId, Offset>
}

impl<'a> QueryVec<'a> {
    pub fn new() -> Self {
        Self {
            queries: VariableWidthSliceVector::new()
        }
    }

    pub fn push(self: &mut Self, query: &[usize]) {
        let word_ids: Vec<_> = query.iter().map(|&x| x.into()).collect();
        self.queries.push(&word_ids);
    }

    pub fn get(self: &Self, idx: usize) -> Vec<usize> {
        self.queries.get(idx).iter().map(|&word_id| word_id.into()).collect()
    }

    pub fn len(self: &Self) -> usize {
        self.queries.len()
    }

    pub fn extend_from_queryvec<'b>(self: &mut Self, other: &QueryVec<'b>) {
        self.queries.extend_from_slice_vector(&other.queries);
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        Self {
            queries: VariableWidthSliceVector::load(buffer)
        }
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        self.queries.write(buffer)
    }
}

pub fn get_random_word_embeddings(dimension: usize, num: usize) -> WordEmbeddings<'static>
{
    let mut rng = rand::thread_rng();

    let embeddings: types::AngularVectors = (0..num).map(|_| {
        let element: AngularVector = (0..dimension).map(|_| rng.gen::<f32>() - 0.5).collect();

        element
    }).collect();

    WordEmbeddings {
        embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use std::io::Write;
    use std::fs::File;
    use memmap;

    #[test]
    fn queryvec_push() {
        let mut qvec = QueryVec::new();

        let example_query: Vec<_> = (0..5).collect();
        for _ in 0..11 {
            qvec.push(&example_query);
        }

        assert_eq!(11, qvec.len());
    }

    #[test]
    fn queryvec_get() {
        let mut qvec = QueryVec::new();

        qvec.push(&[44]);
        let example_query: Vec<_> = (0..5).collect();
        for _ in 1..12 {
            qvec.push(&example_query);
        }

        assert_eq!(vec![44], qvec.get(0));

        for i in 1..12 {
            assert_eq!(example_query, qvec.get(i));
        }
    }

    #[test]
    fn queryvec_extend() {
        let mut qvec = QueryVec::new();
        qvec.push(&[44, 1234]);
        qvec.push(&[132,151]);
        qvec.push(&[0,2]);

        let mut other = QueryVec::new();
        other.push(&[17]);
        other.push(&[9, 12345, 6245]);

        qvec.extend_from_queryvec(&other);

        assert_eq!(5, qvec.len());
        assert_eq!(vec![44,1234], qvec.get(0));
        assert_eq!(vec![132,151], qvec.get(1));
        assert_eq!(vec![0,2], qvec.get(2));
        assert_eq!(vec![17], qvec.get(3));
        assert_eq!(vec![9,12345,6245], qvec.get(4));
    }

    #[test]
    fn queryvec_extend_empty() {
        let mut qvec = QueryVec::new();
        qvec.push(&[44, 1234]);
        qvec.push(&[132,151]);
        qvec.push(&[0,2]);

        qvec.extend_from_queryvec(&QueryVec::new());

        assert_eq!(3, qvec.len());
        assert_eq!(vec![44,1234], qvec.get(0));
        assert_eq!(vec![132,151], qvec.get(1));
        assert_eq!(vec![0,2], qvec.get(2));
    }

    #[test]
    fn queryvec_empty_extend() {
        let mut qvec = QueryVec::new();

        let mut other = QueryVec::new();
        other.push(&[17]);
        other.push(&[9, 12345, 6245]);

        qvec.extend_from_queryvec(&other);

        assert_eq!(2, qvec.len());
        assert_eq!(vec![17], qvec.get(0));
        assert_eq!(vec![9,12345,6245], qvec.get(1));
    }

    #[test]
    fn queryvec_write_and_load() {
        let mut qvec = QueryVec::new();

        for i in 1..12 {
            let query: Vec<_> = (0..i).collect();
            qvec.push(&query);
        }

        let mut buffer = Vec::new();
        qvec.write(&mut buffer).unwrap();

        let qvec2 = QueryVec::load(&buffer);

        assert_eq!(qvec.len(), qvec2.len());

        for i in 0..qvec.len() {
            assert_eq!(qvec.get(i), qvec2.get(i));
        }
    }

    #[test]
    fn test_word_embeddings_write_load() {
        let mut buffer: Vec<u8> = Vec::new();
        let num_word_embeddings = 3;

        const DIM: usize = 300;

        let word_embeddings = get_random_word_embeddings(DIM, num_word_embeddings);

        word_embeddings.embeddings.write(&mut buffer).unwrap();

        let mut temp_file = temp_dir();
        temp_file.push("test_get_word_embeddings");

        {
            let mut file = File::create(&temp_file).unwrap();
            file.write_all(&buffer).unwrap();
        }

        let file = File::open(&temp_file).unwrap();
        let word_embeddings = unsafe { memmap::Mmap::map(&file).unwrap() };

        let we = WordEmbeddings::load(DIM, &word_embeddings);

        assert_eq!(num_word_embeddings, we.embeddings.len());
    }
}
