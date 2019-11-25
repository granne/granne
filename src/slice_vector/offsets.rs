use crate::io;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::borrow::Cow;
use std::io::{Read, Result, Seek, SeekFrom, Write};

pub const OFFSETS_PER_CHUNK: usize = 60;
type DIFF_TYPE = u16;

#[derive(Clone)]
pub struct CompressedVariableWidthSliceVector<'a, T: 'a + Clone> {
    offsets: Offsets<'a>,
    data: Cow<'a, [T]>,
}

impl<'a, T: 'a + Clone> CompressedVariableWidthSliceVector<'a, T> {
    pub fn new() -> Self {
        let mut offsets = Offsets::new();
        offsets.push(0);

        Self {
            offsets,
            data: Vec::new().into(),
        }
    }

    pub fn get<'b>(self: &'b Self, idx: usize) -> &'b [T]
    where
        'a: 'b,
    {
        let (begin, end) = self.offsets.get_consecutive(idx);

        &self.data[begin..end]
    }

    pub fn len(self: &Self) -> usize {
        self.offsets.len() - 1
    }

    pub fn push(self: &mut Self, data: &[T]) {
        self.data.to_mut().extend_from_slice(data);
        self.offsets.push(self.data.len());
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        // write metadata
        buffer
            .write_u64::<LittleEndian>(self.offsets.num_bytes() as u64)
            .expect("Could not write length");

        let mut bytes_written = std::mem::size_of::<u64>();

        bytes_written += self.offsets.write(buffer)?;
        bytes_written += io::write_as_bytes(&self.data[..], buffer)?;

        Ok(bytes_written)
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        let u64_len = ::std::mem::size_of::<u64>();
        let num_bytes = (&buffer[..u64_len])
            .read_u64::<LittleEndian>()
            .expect("Could not read length") as usize;

        let (offsets, data) = buffer[u64_len..].split_at(num_bytes);

        unsafe {
            Self {
                offsets: Offsets::load(offsets),
                data: Cow::from(crate::io::load_bytes_as::<T>(data)),
            }
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Chunk {
    initial: usize,
    deltas: [DIFF_TYPE; OFFSETS_PER_CHUNK],
}

const UNUSED: DIFF_TYPE = DIFF_TYPE::max_value();

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
        for i in 0..(index + 1) {
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
        assert!(delta <= DIFF_TYPE::max_value() as usize);
        self.deltas[self.len()] = delta as DIFF_TYPE;
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
        if self
            .chunks
            .last()
            .expect("Chunks should not be empty()")
            .is_full()
        {
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
