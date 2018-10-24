use bytes::{ByteOrder, LittleEndian};
use file_io;
use hnsw::{At, Writeable};
use std::borrow::Cow;
use std::io::{Result, Write};
use types;
use types::Dense;
use types::AngularVector;
use rand::Rng;
use rand;
use blas;

pub mod parsing;

pub const DIM: usize = 300;

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
        self.queries.get(idx)
    }

    pub fn get_embedding(self: &Self, idx: usize) -> types::AngularVector<'static>
    {
        // save some time by avoiding conversion to Vec<usize> for word ids
        self.word_embeddings.get_embedding_internal(&self.queries.get_ref(idx))
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
struct Offset([u8; BYTES_PER_OFFSET]);

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
struct WordId([u8; BYTES_PER_WORD_ID]);

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
    offsets: Cow<'a, [Offset]>,
    word_ids: Cow<'a, [WordId]>,
}

impl<'a> QueryVec<'a> {
    pub fn new() -> Self {
        Self {
            offsets: Cow::from(vec![0.into()]),
            word_ids: Cow::from(Vec::new())
        }
    }

    pub fn push(self: &mut Self, query: &[usize]) {
        for &w in query {
            self.word_ids.to_mut().push(w.into());
        }

        self.offsets.to_mut().push(self.word_ids.len().into());
    }

    pub fn get(self: &Self, idx: usize) -> Vec<usize> {
        self.get_ref(idx).iter().map(|&word_id| word_id.into()).collect()
    }

    fn get_ref(self: &'a Self, idx: usize) -> &'a [WordId] {
        let start: usize = self.offsets[idx].into();
        let end: usize = self.offsets[idx + 1].into();

        &self.word_ids[start..end]
    }

    pub fn len(self: &Self) -> usize {
        self.offsets.len() - 1
    }

    pub fn extend_from_queryvec(self: &mut Self, other: &QueryVec) {
        let prev_len: usize = (*self.offsets.last().unwrap()).into();

        self.offsets.to_mut().extend(
            other.offsets
                .iter()
                .skip(1) // skip the initial 0
                .map(|&x| {
                    let x: usize = x.into();
                    let x: Offset = (prev_len + x).into();
                    x
                })
        );

        self.word_ids.to_mut().extend_from_slice(&other.word_ids);
    }

    pub fn load(buffer: &'a [u8]) -> Self
    {
        let num_queries = LittleEndian::read_u64(buffer) as usize;
        let (offsets, word_ids) = buffer[::std::mem::size_of::<u64>()..].split_at((1 + num_queries) * BYTES_PER_OFFSET);

        Self {
            offsets: Cow::from(file_io::load::<Offset>(offsets)),
            word_ids: Cow::from(file_io::load::<WordId>(word_ids)),
        }
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        // write metadata
        let mut buf = [0u8; 8];
        LittleEndian::write_u64(&mut buf, self.len() as u64);
        buffer.write_all(&buf)?;

        // write queries
        let offsets = unsafe {
            ::std::slice::from_raw_parts(
                self.offsets.as_ptr() as *const u8,
                self.offsets.len() * ::std::mem::size_of::<Offset>(),
            )
        };

        buffer.write_all(offsets)?;

        let word_ids = unsafe {
            ::std::slice::from_raw_parts(
                self.word_ids.as_ptr() as *const u8,
                self.word_ids.len() * ::std::mem::size_of::<WordId>(),
            )
        };

        buffer.write_all(word_ids)?;

        Ok(())
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
