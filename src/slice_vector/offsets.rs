use crate::io;
use madvise::{AccessPattern, AdviseMemory};
use std::borrow::Cow;
use std::io::{Read, Result, Seek, SeekFrom, Write};

use super::U64_LEN;
pub const OFFSETS_PER_CHUNK: usize = 60;
type DeltaType = u16;

pub enum CompressedVariableWidthSliceVector<'a, T: Clone> {
    File(memmap::Mmap),
    Memory(Offsets<'a>, Cow<'a, [T]>),
}

impl<'a, T: Clone> Clone for CompressedVariableWidthSliceVector<'a, T> {
    fn clone(self: &Self) -> Self {
        match self {
            Self::File(mmap) => {
                let (offsets, data) = Self::load_mmap(&mmap[..]);
                Self::Memory(offsets.into_owned(), Cow::Owned(data.to_vec()))
            }
            Self::Memory(offsets, data) => Self::Memory(offsets.clone(), data.clone()),
        }
    }
}

impl<'a, T: Clone> CompressedVariableWidthSliceVector<'a, T> {
    pub fn new() -> Self {
        let mut offsets = Offsets::new();
        offsets.push(0);

        Self::Memory(offsets, Cow::Owned(Vec::new()))
    }

    pub unsafe fn from_file(file: &std::fs::File) -> std::io::Result<Self> {
        let file = memmap::Mmap::map(&file)?;
        file.advise_memory_access(AccessPattern::Random)?;
        let slice_vec = Self::File(file);

        // try to fail early
        let (_offsets, _data) = slice_vec.load();

        Ok(slice_vec)
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        let (offsets, data) = Self::load_mmap(buffer);
        Self::Memory(offsets, Cow::Borrowed(data))
    }

    pub fn get<'b>(self: &'b Self, idx: usize) -> &'b [T]
    where
        'a: 'b,
    {
        let (offsets, data) = self.load();

        let (begin, end) = offsets.get_consecutive(idx);

        &data[begin..end]
    }

    pub fn len(self: &Self) -> usize {
        let (offsets, _data) = self.load();

        offsets.len() - 1
    }

    pub fn push(self: &mut Self, new_data: &[T]) {
        let (offsets, data) = self.load_mut();

        data.extend_from_slice(new_data);
        offsets.push(data.len());
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        let (offsets, data) = self.load();

        // write metadata
        buffer.write_all(&(offsets.num_bytes() as u64).to_le_bytes())?;

        let mut bytes_written = std::mem::size_of::<u64>();

        bytes_written += offsets.write(buffer)?;
        bytes_written += io::write_as_bytes(&data[..], buffer)?;

        Ok(bytes_written)
    }

    fn load<'b>(self: &'b Self) -> (Offsets<'b>, &'b [T])
    where
        'a: 'b,
    {
        match self {
            Self::File(mmap) => Self::load_mmap(&mmap[..]),
            Self::Memory(offsets, data) => (offsets.borrow(), &data),
        }
    }

    fn load_mut(self: &mut Self) -> (&mut Offsets<'a>, &mut Vec<T>) {
        match self {
            Self::File(mmap) => {
                let (offsets, data) = Self::load_mmap(mmap);
                *self = Self::Memory(offsets.into_owned(), Cow::Owned(data.to_vec()));
            }
            Self::Memory(_, _) => {}
        }

        match self {
            Self::File(_) => unreachable!(),
            Self::Memory(ref mut offsets, data) => (offsets, data.to_mut()),
        }
    }

    fn load_mmap(buffer: &[u8]) -> (Offsets<'_>, &[T]) {
        let num_bytes = {
            let mut buf = [0x0; U64_LEN];
            buf.copy_from_slice(&buffer[..U64_LEN]);
            u64::from_le_bytes(buf) as usize
        };

        let (offsets, data) = buffer[U64_LEN..].split_at(num_bytes);

        (Offsets::load(offsets), unsafe { crate::io::load_bytes_as::<T>(data) })
    }

    pub fn borrow<'b>(self: &'a Self) -> CompressedVariableWidthSliceVector<'b, T>
    where
        'a: 'b,
    {
        let (offsets, data) = self.load();

        Self::Memory(offsets, Cow::Borrowed(data))
    }

    pub fn into_owned(self: Self) -> CompressedVariableWidthSliceVector<'static, T> {
        match self {
            Self::File(mmap) => {
                let (offsets, data) = Self::load_mmap(&mmap[..]);
                CompressedVariableWidthSliceVector::Memory(offsets.into_owned(), Cow::Owned(data.to_vec()))
            }
            Self::Memory(offsets, data) => {
                CompressedVariableWidthSliceVector::Memory(offsets.into_owned(), Cow::Owned(data.into_owned()))
            }
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Chunk {
    initial: usize,
    deltas: [DeltaType; OFFSETS_PER_CHUNK],
}

const UNUSED: DeltaType = DeltaType::max_value();

impl Chunk {
    fn new(initial: usize) -> Self {
        Self {
            initial,
            deltas: [UNUSED; OFFSETS_PER_CHUNK],
        }
    }

    fn len(self: &Self) -> usize {
        self.deltas.iter().take_while(|&&x| x != UNUSED).count()
    }

    fn is_empty(self: &Self) -> bool {
        self.deltas[0] == UNUSED
    }

    fn is_full(self: &Self) -> bool {
        self.deltas[OFFSETS_PER_CHUNK - 1] != UNUSED
    }

    #[inline(always)]
    fn get(self: &Self, index: usize) -> usize {
        assert!(index < OFFSETS_PER_CHUNK);
        let mut res = 0;
        for i in 0..=index {
            debug_assert!(UNUSED != self.deltas[i]);
            res += self.deltas[i] as usize;
        }

        res + self.initial
    }

    fn get_consecutive(self: &Self, index: usize) -> (usize, usize) {
        assert!(index + 1 < OFFSETS_PER_CHUNK);
        let first = self.get(index);
        let second = first + self.deltas[index + 1] as usize;

        (first, second)
    }

    fn push(self: &mut Self, offset: usize) {
        assert!(!self.is_full());
        let last_offset = if let Some(last) = self.last() {
            last
        } else {
            self.initial
        };

        assert!(offset >= last_offset);
        let delta = offset - last_offset;
        assert!(delta <= DeltaType::max_value() as usize);
        self.deltas[self.len()] = delta as DeltaType;
    }

    fn last(self: &Self) -> Option<usize> {
        if !self.is_empty() {
            Some(self.get(self.len() - 1))
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Offsets<'a> {
    chunks: Cow<'a, [Chunk]>,
}

impl<'a> Offsets<'a> {
    pub fn new() -> Self {
        Self {
            chunks: Cow::Owned(vec![Chunk::new(0)]),
        }
    }

    fn from_parts(chunks: &'a [Chunk]) -> Self {
        Self {
            chunks: Cow::Borrowed(chunks),
        }
    }

    pub fn push(self: &mut Self, offset: usize) {
        if self.chunks.last().expect("Chunks should not be empty()").is_full() {
            let mut new_chunk = Chunk::new(offset);
            new_chunk.push(offset);
            self.chunks.to_mut().push(new_chunk);
        } else {
            self.chunks.to_mut().last_mut().unwrap().push(offset);
        }
    }

    #[inline(always)]
    pub fn get(self: &Self, index: usize) -> usize {
        self.chunks[index / OFFSETS_PER_CHUNK].get(index % OFFSETS_PER_CHUNK)
    }

    pub fn get_consecutive(self: &Self, index: usize) -> (usize, usize) {
        if (index + 1) % OFFSETS_PER_CHUNK == 0 {
            (self.get(index), self.get(index + 1))
        } else {
            self.chunks[index / OFFSETS_PER_CHUNK].get_consecutive(index % OFFSETS_PER_CHUNK)
        }
    }

    pub fn last(self: &Self) -> Option<usize> {
        self.chunks.last()?.last()
    }

    #[inline(always)]
    pub fn len(self: &Self) -> usize {
        OFFSETS_PER_CHUNK * (self.chunks.len() - 1) + self.chunks.last().unwrap().len()
    }

    pub fn num_bytes(self: &Self) -> usize {
        self.chunks.len() * ::std::mem::size_of::<Chunk>()
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        crate::io::write_as_bytes(&self.chunks[..], buffer)
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        Offsets::from_parts(unsafe { crate::io::load_bytes_as::<Chunk>(buffer) })
    }

    pub fn borrow<'b>(self: &'a Self) -> Offsets<'b>
    where
        'a: 'b,
    {
        Self {
            chunks: Cow::Borrowed(&self.chunks),
        }
    }

    pub fn into_owned(self: Self) -> Offsets<'static> {
        Offsets {
            chunks: Cow::Owned(self.chunks.into_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    #[test]
    fn general_behavior() {
        let mut reference = Vec::new();
        let mut offsets = Offsets::new();

        offsets.push(9);
        reference.push(9);

        for i in 0..255 {
            offsets.push(reference.last().unwrap() + i);
            reference.push(reference.last().unwrap() + i);
        }

        assert_eq!(reference.len(), offsets.len());

        for i in 0..reference.len() {
            assert_eq!(reference[i], offsets.get(i));
        }
    }

    #[test]
    fn last() {
        let mut offsets = Offsets::new();
        for offset in test_helper::random_offsets(u16::max_value() as usize).take(1000) {
            offsets.push(offset);
            assert_eq!(offset, offsets.last().unwrap());
        }
    }

    #[test]
    fn empty() {
        let mut offsets = Offsets::new();
        assert_eq!(0, offsets.len());

        offsets.push(0);
        for i in 0..100 {
            offsets.push(0);
            assert_eq!(0, offsets.get(i));
        }
    }

    #[test]
    #[should_panic]
    fn not_offset() {
        let mut offsets = Offsets::new();

        offsets.push(14);
        offsets.push(5);
    }
}
