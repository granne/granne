#![allow(unused)]

use rayon::prelude::ParallelSliceMut;
use rayon::prelude::*;

use std::borrow::Cow;
use std::convert::TryFrom;
use std::io::{Read, Result, Seek, SeekFrom, Write};
mod offsets;
mod set_vector;

use madvise::{AccessPattern, AdviseMemory};
use memmap;

use crate::io;

pub use offsets::{Chunk, CompressedVariableWidthSliceVector, Offsets};

pub use set_vector::MultiSetVector;

const U64_LEN: usize = ::std::mem::size_of::<u64>();

/// A vector containing variably wide slices.
pub enum VariableWidthSliceVector<'a, T: 'a + Clone, Offset: 'a + Clone> {
    File(memmap::Mmap),
    Memory(Cow<'a, [Offset]>, Cow<'a, [T]>),
}

impl<'a, T: Clone, Offset: Clone> Clone for VariableWidthSliceVector<'a, T, Offset> {
    fn clone(self: &Self) -> Self {
        match self {
            Self::File(mmap) => {
                let (offsets, data) = Self::load_mmap(&mmap[..]);
                Self::Memory(Cow::Owned(offsets.to_vec()), Cow::Owned(data.to_vec()))
            }
            Self::Memory(offsets, data) => Self::Memory(offsets.clone(), data.clone()),
        }
    }
}

/// A vector containing fixed width slices.
pub enum FixedWidthSliceVector<'a, T: Clone> {
    File(memmap::Mmap),
    Memory(Cow<'a, [T]>, usize),
}

impl<'a, T: Clone> Clone for FixedWidthSliceVector<'a, T> {
    fn clone(self: &Self) -> Self {
        match self {
            Self::File(mmap) => {
                let (data, width) = Self::load_mmap(&mmap[..]);
                Self::Memory(Cow::Owned(data.to_vec()), width)
            }
            Self::Memory(data, width) => Self::Memory(data.clone(), *width),
        }
    }
}

impl<'a, T: Clone> Default for FixedWidthSliceVector<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Clone> Into<Vec<T>> for FixedWidthSliceVector<'a, T> {
    fn into(self: Self) -> Vec<T> {
        match self {
            Self::File(mmap) => {
                let (data, _width) = Self::load_mmap(&mmap[..]);
                data.to_vec()
            }
            Self::Memory(data, _width) => data.into_owned(),
        }
    }
}

impl<'a, T: Clone> FixedWidthSliceVector<'a, T> {
    /// Creates an empty FixedWidthSliceVector with unspecified width.
    pub fn new() -> Self {
        Self::with_width(0)
    }

    /// Creates an empty FixedWidthSliceVector with `width`.
    pub fn with_width(width: usize) -> Self {
        Self::Memory(Vec::new().into(), width)
    }

    /// Creates an empty FixedWidthSliceVector with space for at least `capacity` slices with
    /// `width`.
    ///
    /// `width` must be greater than 0.
    pub fn with_capacity(width: usize, capacity: usize) -> Self {
        assert!(width > 0);

        Self::Memory(Vec::with_capacity(width * capacity).into(), width)
    }

    /// Creates a FixedWidthSliceVector with data from `data`
    ///
    /// `width` must be greater than 0 and `data.len()` must be divisible by `width`.
    pub fn with_data(data: impl Into<Cow<'a, [T]>>, width: usize) -> Self {
        let data = data.into();

        assert!(width > 0);
        assert!(data.len() % width == 0);

        Self::Memory(data, width)
    }

    pub unsafe fn from_file(file: &std::fs::File) -> std::io::Result<Self> {
        let file = memmap::Mmap::map(&file)?;
        file.advise_memory_access(AccessPattern::Random)?;
        let slice_vec = Self::File(file);

        // try to fail early
        let (data, width) = slice_vec.load();

        assert!(width > 0 && data.len() % width == 0);

        Ok(slice_vec)
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        let (data, width) = Self::load_mmap(buffer);

        assert!(width > 0 && data.len() % width == 0);

        Self::Memory(Cow::Borrowed(data), width)
    }

    /// Returns an iterator over all slices in `self`.
    pub fn iter<'b>(self: &'b Self) -> impl Iterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        let (data, width) = self.load();
        data.chunks(width)
    }

    /// Returns an iterator over mutable slices in `self`.
    pub fn iter_mut<'b>(self: &'b mut Self) -> impl Iterator<Item = &'b mut [T]>
    where
        'a: 'b,
    {
        let (data, width) = self.load_mut();
        data.chunks_mut(*width)
    }

    /// Creates a new FixedWidthSliceVector containing all slices in `self` between `begin` and
    /// `end`.
    pub fn subslice(self: &'_ Self, begin: usize, end: usize) -> FixedWidthSliceVector<'_, T> {
        let (data, width) = self.load();

        let begin = begin * width;
        let end = end * width;

        FixedWidthSliceVector::with_data(&data[begin..end], width)
    }

    pub fn reserve(self: &mut Self, additional: usize) {
        let (data, width) = self.load_mut();
        data.reserve(additional * *width);
    }

    pub fn reserve_exact(self: &mut Self, additional: usize) {
        let (data, width) = self.load_mut();
        data.reserve_exact(additional * *width);
    }

    pub fn resize(self: &mut Self, new_len: usize, value: T) {
        let (data, width) = self.load_mut();
        data.resize(new_len * *width, value);
    }

    /// Extend this FixedWidthSliceVector with the slices from `other`.
    ///
    /// The widths of `self` and `other` must be equal.
    pub fn extend_from_slice_vector(self: &mut Self, other: &FixedWidthSliceVector<T>) {
        let (data, width) = self.load_mut();
        let (other_data, other_width) = other.load();

        if *width == 0 {
            *width = other_width;
        }

        assert_eq!(*width, other_width);

        data.extend_from_slice(other_data);
    }

    fn load_mut(self: &mut Self) -> (&mut Vec<T>, &mut usize) {
        match self {
            Self::File(mmap) => {
                let (data, width) = Self::load_mmap(&mmap[..]);
                *self = Self::Memory(Cow::Owned(data.to_vec()), width)
            }
            Self::Memory(_, _) => {}
        }

        match self {
            Self::File(_) => unreachable!(),
            Self::Memory(data, width) => (data.to_mut(), width),
        }
    }

    fn load(self: &Self) -> (&[T], usize) {
        match self {
            Self::File(mmap) => Self::load_mmap(&mmap[..]),
            Self::Memory(data, width) => (&data, *width),
        }
    }

    fn load_mmap(buffer: &[u8]) -> (&[T], usize) {
        let width = {
            let mut buf = [0x0; U64_LEN];
            buf.copy_from_slice(&buffer[..U64_LEN]);
            u64::from_le_bytes(buf) as usize
        };

        (unsafe { crate::io::load_bytes_as(&buffer[U64_LEN..]) }, width)
    }

    pub fn read<I: Read>(mut reader: I) -> Result<Self> {
        let width = {
            let mut buf = [0x0; U64_LEN];
            reader.read_exact(&mut buf);
            u64::from_le_bytes(buf) as usize
        };

        let mut buffer = Vec::new();
        buffer.resize(width * ::std::mem::size_of::<T>(), 0);

        let mut vec = Self::new();

        while let Ok(()) = reader.read_exact(&mut buffer) {
            vec.push(unsafe { crate::io::load_bytes_as(&buffer[..]) });
        }

        Ok(vec)
    }

    pub fn read_with_capacity<I: Read>(mut reader: I, capacity: usize) -> Result<Self> {
        let width = {
            let mut buf = [0x0; U64_LEN];
            reader.read_exact(&mut buf);
            u64::from_le_bytes(buf) as usize
        };

        let mut buffer = Vec::new();
        buffer.resize(width * ::std::mem::size_of::<T>(), 0);

        let mut vec = Self::with_capacity(width, capacity);

        while let Ok(()) = reader.read_exact(&mut buffer) {
            vec.push(unsafe { crate::io::load_bytes_as(&buffer[..]) });
        }

        Ok(vec)
    }

    pub fn write_as_variable_width_slice_vector<Offset, B, P>(
        self: &Self,
        buffer: &mut B,
        mut predicate: P,
    ) -> Result<usize>
    where
        Offset: TryFrom<usize>,
        <Offset as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
        B: Write + Seek,
        P: FnMut(&T) -> bool,
    {
        let initial_pos = buffer.seek(SeekFrom::Current(0))?;

        buffer.write_all(&(self.len() as u64).to_le_bytes())?;

        let zero_offset = Offset::try_from(0).unwrap();
        io::write_as_bytes(&[zero_offset], buffer)?;
        let mut offset_pos = buffer.seek(SeekFrom::Current(0))?;
        let offset_size = ::std::mem::size_of::<Offset>() as u64;
        let mut value_pos = offset_pos + self.len() as u64 * offset_size;

        let mut slice_buffer: Vec<T> = Vec::new();
        let mut offsets: Vec<Offset> = Vec::new();

        // write to file in chunks (better for BufWriter)
        let mut num_chunks = 100;
        let chunk_size = std::cmp::max(100, self.len() / num_chunks);
        num_chunks = (self.len() + chunk_size - 1) / chunk_size;

        let mut total_count: usize = 0;
        for chunk in 0..num_chunks {
            // starting index for this chunk
            let chunk_offset = chunk * chunk_size;

            // chunk_size or whatever is left
            let chunk_size = std::cmp::min(chunk_size, self.len() - chunk_offset);

            offsets.clear();
            buffer.seek(SeekFrom::Start(value_pos))?;
            for i in 0..chunk_size {
                slice_buffer.clear();
                for val in self.get(chunk_offset + i) {
                    if predicate(val) {
                        slice_buffer.push(val.clone());
                    }
                }
                total_count += slice_buffer.len();
                io::write_as_bytes(slice_buffer.as_slice(), buffer)?;
                offsets.push(Offset::try_from(total_count).unwrap());
            }
            value_pos = buffer.seek(SeekFrom::Current(0))?;

            buffer.seek(SeekFrom::Start(offset_pos))?;
            io::write_as_bytes(offsets.as_slice(), buffer)?;
            offset_pos = buffer.seek(SeekFrom::Current(0))?;
        }

        buffer.seek(SeekFrom::Start(value_pos))?;

        let bytes_written = (value_pos - initial_pos) as usize;

        Ok(bytes_written)
    }

    pub fn borrow<'b>(self: &'a Self) -> FixedWidthSliceVector<'b, T>
    where
        'a: 'b,
    {
        let (data, width) = self.load();

        Self::Memory(Cow::Borrowed(data), width)
    }

    pub fn into_owned(self: Self) -> FixedWidthSliceVector<'static, T> {
        match self {
            Self::File(mmap) => {
                let (data, width) = Self::load_mmap(&mmap[..]);
                FixedWidthSliceVector::Memory(Cow::Owned(data.to_vec()), width)
            }
            Self::Memory(data, width) => FixedWidthSliceVector::Memory(Cow::Owned(data.into_owned()), width),
        }
    }

    pub fn get<'b>(self: &'b Self, idx: usize) -> &'b [T]
    where
        'a: 'b,
    {
        let (data, width) = self.load();

        debug_assert!(idx < self.len());

        let begin = idx * width;
        let end = (idx + 1) * width;

        &data[begin..end]
    }

    pub fn get_mut<'b>(self: &'b mut Self, idx: usize) -> &'b mut [T]
    where
        'a: 'b,
    {
        debug_assert!(idx < self.len());

        let (data, width) = self.load_mut();

        let begin = idx * *width;
        let end = begin + *width;

        &mut data[begin..end]
    }

    fn get_two_mut<'b>(self: &'b mut Self, idx0: usize, idx1: usize) -> (&'b mut [T], &'b mut [T])
    where
        'a: 'b,
    {
        assert!(idx0 != idx1);

        let (data, width) = self.load_mut();
        let width = *width;

        if idx0 < idx1 {
            let (left, right) = data.split_at_mut(idx1 * width);

            let begin = idx0 * width;
            let end = begin + width;

            (&mut left[begin..end], &mut right[..width])
        } else {
            let (left, right) = data.split_at_mut(idx0 * width);

            let begin = idx1 * width;
            let end = begin + width;

            (&mut right[..width], &mut left[begin..end])
        }
    }

    pub fn push(self: &mut Self, new_data: &[T]) {
        let (data, width) = self.load_mut();

        if *width == 0 {
            *width = new_data.len();
        }

        assert_eq!(*width, new_data.len());

        data.extend_from_slice(new_data);
    }

    pub fn len(self: &Self) -> usize {
        let (data, width) = self.load();
        if width > 0 {
            data.len() / width
        } else {
            0
        }
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }

    pub fn width(self: &Self) -> usize {
        let (_, width) = self.load();
        width
    }

    pub fn as_slice(self: &Self) -> &[T] {
        let (data, _width) = self.load();
        &data[..]
    }

    /// Permutes this vector by `permutation`. The element at index `permutation[i]` will
    /// move to index `i`.
    ///
    /// E.g. `permutation = [1, 2, 0]` permutes `['a', 'b', 'c']` into `['b', 'c', 'a'].
    pub fn permute(self: &mut Self, permutation: &[usize]) {
        assert_eq!(self.len(), permutation.len());

        let mut visited = vec![false; self.len()];

        for i in 0..permutation.len() {
            if visited[i] {
                continue;
            }

            let loop_begin = self.get(i).to_vec();
            let mut j = i;
            while permutation[j] != i {
                let (to, from) = self.get_two_mut(j, permutation[j]);
                to.clone_from_slice(from);
                visited[j] = true;
                j = permutation[j];
            }
            self.get_mut(j).clone_from_slice(&loop_begin);
            visited[j] = true;
        }
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        let (data, width) = self.load();

        buffer.write_all(&width.to_le_bytes())?;

        io::write_as_bytes(&data[..], buffer)
    }
}

impl<'a, T: 'a + Clone + Send + Sync> FixedWidthSliceVector<'a, T> {
    pub fn par_iter<'b>(self: &'b Self) -> impl IndexedParallelIterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        let (data, width) = self.load();
        data.par_chunks(width)
    }

    pub fn par_iter_mut<'b>(self: &'b mut Self) -> impl IndexedParallelIterator<Item = &'b mut [T]>
    where
        'a: 'b,
    {
        let (data, width) = self.load_mut();
        data.par_chunks_mut(*width)
    }
}

impl<'a, T: Clone, Offset: TryFrom<usize> + Copy> Default for VariableWidthSliceVector<'a, T, Offset>
where
    usize: TryFrom<Offset>,
    <Offset as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    <usize as std::convert::TryFrom<Offset>>::Error: std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Clone, Offset: TryFrom<usize> + Copy> VariableWidthSliceVector<'a, T, Offset>
where
    usize: TryFrom<Offset>,
    <Offset as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    <usize as std::convert::TryFrom<Offset>>::Error: std::fmt::Debug,
{
    pub fn new() -> Self {
        Self::Memory(vec![Offset::try_from(0).unwrap()].into(), Vec::new().into())
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
        Self::Memory(Cow::Borrowed(offsets), Cow::Borrowed(data))
    }

    pub fn extend_from_slice_vector(self: &mut Self, other: &VariableWidthSliceVector<T, Offset>) {
        let (offsets, data) = self.load_mut();
        let (other_offsets, other_data) = other.load();

        let prev_len = usize::try_from(*offsets.last().unwrap()).unwrap();

        offsets.extend(
            other_offsets
                .iter()
                .skip(1) // skip the initial 0
                .map(|&x| {
                    let x = usize::try_from(x).unwrap();
                    Offset::try_from(prev_len + x).unwrap()
                }),
        );

        data.extend_from_slice(other_data);
    }

    pub fn iter<'b>(self: &'b Self) -> impl Iterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        let (offsets, data) = self.load();

        offsets.iter().zip(offsets.iter().skip(1)).map(move |(&begin, &end)| {
            let begin = usize::try_from(begin).unwrap();
            let end = usize::try_from(end).unwrap();

            &data[begin..end]
        })
    }

    pub fn borrow<'b>(self: &'a Self) -> VariableWidthSliceVector<'b, T, Offset>
    where
        'a: 'b,
    {
        let (offsets, data) = self.load();
        Self::Memory(Cow::Borrowed(offsets), Cow::Borrowed(data))
    }

    pub fn write_range<B: Write>(self: &Self, buffer: &mut B, begin: usize, end: usize) -> Result<usize> {
        assert!(begin <= end);
        assert!(begin <= self.len());
        assert!(end <= self.len());

        let (offsets, data) = self.load();

        // write metadata
        buffer.write_all(&((end - begin) as u64).to_le_bytes())?;
        let mut bytes_written = std::mem::size_of::<u64>();

        let offset_begin = usize::try_from(offsets[begin]).unwrap();

        for &offset in &offsets[begin..=end] {
            let offset = usize::try_from(offset).unwrap();
            let offset = Offset::try_from(offset - offset_begin).unwrap();
            io::write_as_bytes(&[offset], buffer)?;
        }

        let data_begin = usize::try_from(offsets[begin]).unwrap();
        let data_end = usize::try_from(offsets[end]).unwrap();
        io::write_as_bytes(&data[data_begin..data_end], buffer)
    }

    pub fn get(self: &Self, idx: usize) -> &[T] {
        let (offsets, data) = self.load();

        let begin = usize::try_from(offsets[idx]).unwrap();
        let end = usize::try_from(offsets[idx + 1]).unwrap();

        &data[begin..end]
    }

    pub fn get_mut(self: &mut Self, idx: usize) -> &mut [T] {
        let (offsets, data) = self.load_mut();

        let begin = usize::try_from(offsets[idx]).unwrap();
        let end = usize::try_from(offsets[idx + 1]).unwrap();

        &mut data[begin..end]
    }

    pub fn len(self: &Self) -> usize {
        let (offsets, _data) = self.load();
        offsets.len() - 1
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }

    pub fn push(self: &mut Self, new_data: &[T]) {
        let (offsets, data) = self.load_mut();
        data.extend_from_slice(new_data);
        offsets.push(Offset::try_from(data.len()).unwrap());
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        let (offsets, data) = self.load();

        // write metadata
        buffer.write_all(&self.len().to_le_bytes())?;
        let mut bytes_written = std::mem::size_of::<u64>();

        bytes_written += io::write_as_bytes(&offsets[..], buffer)?;
        bytes_written += io::write_as_bytes(&data[..], buffer)?;

        Ok(bytes_written)
    }
}

impl<'a, T: Clone, Offset: Clone> VariableWidthSliceVector<'a, T, Offset> {
    fn load(self: &Self) -> (&[Offset], &[T]) {
        match self {
            Self::File(mmap) => Self::load_mmap(&mmap[..]),
            Self::Memory(offsets, data) => (&offsets, &data),
        }
    }

    fn load_mut(self: &mut Self) -> (&mut Vec<Offset>, &mut Vec<T>) {
        match self {
            Self::File(mmap) => {
                let (offsets, data) = Self::load_mmap(mmap);
                *self = Self::Memory(Cow::Owned(offsets.to_vec()), Cow::Owned(data.to_vec()));
            }
            Self::Memory(_, _) => {}
        }

        match self {
            Self::File(_) => unreachable!(),
            Self::Memory(offsets, data) => (offsets.to_mut(), data.to_mut()),
        }
    }

    fn load_mmap(buffer: &[u8]) -> (&[Offset], &[T]) {
        let num_slices = {
            let mut buf = [0x0; U64_LEN];
            buf.copy_from_slice(&buffer[..U64_LEN]);
            u64::from_le_bytes(buf) as usize
        };

        let offset_len = ::std::mem::size_of::<Offset>();
        let (offsets, data) = buffer[U64_LEN..].split_at((1 + num_slices) * offset_len);

        unsafe {
            (
                crate::io::load_bytes_as::<Offset>(offsets),
                crate::io::load_bytes_as::<T>(data),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use std::fs::File;
    use tempfile;

    #[test]
    fn fixed_width_push() {
        let vec = FixedWidthSliceVector::new();
        let data: Vec<Vec<usize>> = (0..20).map(|i| (i..).take(5).collect()).collect();

        assert_eq!(0, vec.len());

        let mut vec = vec;
        for d in data.iter() {
            vec.push(&d[..]);
        }

        assert_eq!(data.len(), vec.len());

        for (i, d) in data.iter().enumerate() {
            assert_eq!(&d[..], vec.get(i));
        }
    }

    #[test]
    fn variable_width_push() {
        let vec = VariableWidthSliceVector::<_, usize>::new();
        let data: Vec<Vec<usize>> = (0..20).map(|i| (i..).take(1 + i % 7).collect()).collect();

        assert_eq!(0, vec.len());

        let mut vec = vec;
        for d in data.iter() {
            vec.push(&d[..]);
        }

        assert_eq!(data.len(), vec.len());

        for (i, d) in data.iter().enumerate() {
            assert_eq!(&d[..], vec.get(i));
        }
    }

    #[test]
    fn fixed_width_extend() {
        let mut vec0 = FixedWidthSliceVector::new();
        let mut vec1 = FixedWidthSliceVector::new();
        for i in 0..100 {
            let data: Vec<i32> = (i..).take(7).collect();
            if i < 24 {
                vec0.push(&data);
            } else {
                vec1.push(&data);
            }
        }

        test_extend(vec0, vec1);
    }

    #[test]
    #[should_panic]
    fn fixed_width_extend_different_widths() {
        let mut vec0 = FixedWidthSliceVector::with_width(7);
        let mut vec1 = FixedWidthSliceVector::with_width(6);
        for i in 0..100 {
            if i < 24 {
                let data: Vec<i32> = (i..).take(7).collect();
                vec0.push(&data);
            } else {
                let data: Vec<i32> = (i..).take(6).collect();
                vec1.push(&data);
            }
        }

        test_extend(vec0, vec1);
    }

    fn test_extend(vec0: FixedWidthSliceVector<i32>, vec1: FixedWidthSliceVector<i32>) {
        let mut vec_combined = vec0.clone();

        vec_combined.extend_from_slice_vector(&vec1);

        assert_eq!(vec0.len() + vec1.len(), vec_combined.len());

        for i in 0..vec0.len() {
            assert_eq!(vec0.get(i), vec_combined.get(i));
        }

        for i in 0..vec1.len() {
            assert_eq!(vec1.get(i), vec_combined.get(vec0.len() + i));
        }
    }

    #[test]
    fn variable_width_extend() {
        let mut vec0 = VariableWidthSliceVector::<i32, usize>::new();
        let mut vec1 = VariableWidthSliceVector::<i32, usize>::new();
        for i in 0..100 {
            let data: Vec<i32> = (i..).take(1 + (i % 3) as usize).collect();
            if i < 24 {
                vec0.push(&data);
            } else {
                vec1.push(&data);
            }
        }

        test_extend_var(vec0, vec1);
    }

    #[test]
    fn variable_width_extend_empty_left() {
        let vec0 = VariableWidthSliceVector::<i32, usize>::new();
        let mut vec1 = VariableWidthSliceVector::<i32, usize>::new();
        for i in 0..100 {
            let data: Vec<i32> = (i..).take(1 + (i % 3) as usize).collect();
            vec1.push(&data);
        }

        test_extend_var(vec0, vec1);
    }

    #[test]
    fn variable_width_extend_empty_right() {
        let mut vec0 = VariableWidthSliceVector::<i32, usize>::new();
        let vec1 = VariableWidthSliceVector::<i32, usize>::new();
        for i in 0..100 {
            let data: Vec<i32> = (i..).take(1 + (i % 3) as usize).collect();
            vec0.push(&data);
        }

        test_extend_var(vec0, vec1);
    }

    fn test_extend_var(vec0: VariableWidthSliceVector<i32, usize>, vec1: VariableWidthSliceVector<i32, usize>) {
        let mut vec_combined = vec0.clone();

        vec_combined.extend_from_slice_vector(&vec1);

        assert_eq!(vec0.len() + vec1.len(), vec_combined.len());

        for i in 0..vec0.len() {
            assert_eq!(vec0.get(i), vec_combined.get(i));
        }

        for i in 0..vec1.len() {
            assert_eq!(vec1.get(i), vec_combined.get(vec0.len() + i));
        }
    }

    #[test]
    fn fixed_width_write_and_load() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..123 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut buffer = Vec::new();
        vec.write(&mut buffer).unwrap();

        let loaded_vec = FixedWidthSliceVector::<i16>::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn fixed_width_write_and_read() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..123 {
            let data: Vec<i16> = (2 * i + 3..).take(7).collect();
            vec.push(&data);
        }

        let mut buffer = Vec::new();
        vec.write(&mut buffer).unwrap();

        let read_vec = FixedWidthSliceVector::<i16>::read(&mut buffer.as_slice()).unwrap();

        assert_eq!(vec.len(), read_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), read_vec.get(i));
        }
    }

    #[test]
    fn variable_width_write_and_load() {
        let mut vec = VariableWidthSliceVector::<usize, usize>::new();
        for i in 0..19 {
            let data: Vec<usize> = (i..).take(1 + (i % 3) as usize).collect();
            vec.push(&data);
        }

        let mut buffer = Vec::new();
        vec.write(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<usize, usize>::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..123 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        type Offset = usize;

        let mut file: File = tempfile::tempfile().unwrap();
        let bytes_written = vec
            .write_as_variable_width_slice_vector::<Offset, _, _>(&mut file, |_| true)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        use std::mem::size_of;
        assert_eq!(
            size_of::<usize>() + (1 + vec.len()) * size_of::<Offset>() + vec.len() * width * size_of::<i16>(),
            bytes_written
        );

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_predicate() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..522 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        type Offset = usize;

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_variable_width_slice_vector::<Offset, _, _>(&mut file, |x| x % 3 == 0)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            let vec_slice: Vec<_> = vec.get(i).iter().filter(|&x| x % 3 == 0).map(|x| *x).collect();
            assert_eq!(vec_slice.as_slice(), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_empty() {
        let width = 7;
        let vec = FixedWidthSliceVector::<i16>::new();

        type Offset = usize;

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_variable_width_slice_vector::<Offset, _, _>(&mut file, |_| true)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::from_bytes(&buffer);

        assert_eq!(0, loaded_vec.len());
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_empty_slices() {
        let width = 1;
        let mut vec = FixedWidthSliceVector::<i16>::new();

        for _ in 0..10 {
            vec.push(&[0]);
        }

        type Offset = usize;

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_variable_width_slice_vector::<Offset, _, _>(&mut file, |&x| x > 0)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..loaded_vec.len() {
            assert!(loaded_vec.get(i).is_empty());
        }
    }

    #[test]
    fn variable_width_write_range_and_load() {
        let mut vec = VariableWidthSliceVector::<usize, usize>::new();
        for i in 0..19 {
            let data: Vec<usize> = (i..).take(1 + (i % 3) as usize).collect();
            vec.push(&data);
        }

        for begin in 0..vec.len() {
            for end in begin..vec.len() {
                let mut buffer = Vec::new();
                vec.write_range(&mut buffer, begin, end).unwrap();

                let loaded_vec = VariableWidthSliceVector::<usize, usize>::from_bytes(&buffer);

                assert_eq!(end - begin, loaded_vec.len());

                for i in begin..end {
                    assert_eq!(vec.get(i), loaded_vec.get(i - begin));
                }
            }
        }
    }

    #[test]
    fn permute_fixed_width_identity() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..522 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let permutation: Vec<usize> = (0..vec.len()).collect();

        let exp = vec.clone();
        vec.permute(&permutation);

        assert_eq!(exp.len(), vec.len());

        for i in 0..exp.len() {
            assert_eq!(exp.get(i), vec.get(i));
        }
    }

    #[test]
    fn permute_fixed_width_reverse() {
        let width = 5;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..522 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let permutation: Vec<usize> = (0..vec.len()).rev().collect();

        let exp = vec.clone();
        vec.permute(&permutation);

        assert_eq!(exp.len(), vec.len());

        for i in 0..exp.len() {
            assert_eq!(exp.get(i), vec.get(vec.len() - i - 1));
        }
    }

    #[test]
    fn permute_fixed_width_rand_shuffle() {
        let width = 3;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..522 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut rng = rand::thread_rng();
        let mut permutation: Vec<usize> = (0..vec.len()).collect();
        permutation.shuffle(&mut rng);

        let exp = vec.clone();

        vec.permute(&permutation);

        assert_eq!(exp.len(), vec.len());

        for i in 0..exp.len() {
            assert_eq!(exp.get(permutation[i]), vec.get(i));
        }
    }
}
