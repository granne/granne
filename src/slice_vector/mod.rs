#![allow(unused)]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use rayon::prelude::ParallelSliceMut;
use rayon::prelude::*;

use std::borrow::Cow;
use std::convert::TryFrom;
use std::io::{Read, Result, Seek, SeekFrom, Write};
//mod set_vector;

//pub use crate::set_vector::{MultiSetVector, MultiSetVectorT};

#[derive(Clone)]
pub struct VariableWidthSliceVector<'a, T: 'a + Clone, Offset: 'a + Clone> {
    offsets: Cow<'a, [Offset]>,
    data: Cow<'a, [T]>,
}

#[derive(Clone)]
pub struct FixedWidthSliceVector<'a, T: 'a + Clone> {
    data: Cow<'a, [T]>,
    width: usize,
}

impl<'a, T: Clone> Into<Vec<T>> for FixedWidthSliceVector<'a, T> {
    fn into(self: Self) -> Vec<T> {
        self.data.into_owned()
    }
}

impl<'a, T: 'a + Clone> FixedWidthSliceVector<'a, T> {
    pub fn new(width: usize) -> Self {
        assert!(width > 0);

        Self {
            data: Vec::new().into(),
            width,
        }
    }

    pub fn with_capacity(width: usize, capacity: usize) -> Self {
        assert!(width > 0);

        Self {
            data: Vec::with_capacity(width * capacity).into(),
            width,
        }
    }

    pub fn with_data(data: impl Into<Cow<'a, [T]>>, width: usize) -> Self {
        let data = data.into();

        assert!(width > 0);
        assert!(data.len() % width == 0);

        Self { data, width }
    }

    pub fn iter<'b>(self: &'b Self) -> impl Iterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        self.data.chunks(self.width)
    }

    pub fn iter_mut<'b>(self: &'b mut Self) -> impl Iterator<Item = &'b mut [T]>
    where
        'a: 'b,
    {
        self.data.to_mut().chunks_mut(self.width)
    }

    pub fn subslice(self: &'_ Self, begin: usize, end: usize) -> FixedWidthSliceVector<'_, T> {
        let begin = begin * self.width;
        let end = end * self.width;

        FixedWidthSliceVector::with_data(&self.data[begin..end], self.width)
    }

    pub fn reserve(self: &mut Self, additional: usize) {
        self.data.to_mut().reserve(additional * self.width);
    }

    pub fn reserve_exact(self: &mut Self, additional: usize) {
        self.data.to_mut().reserve_exact(additional * self.width);
    }

    pub fn resize(self: &mut Self, new_len: usize, value: T) {
        self.data.to_mut().resize(new_len * self.width, value);
    }

    pub fn extend_from_slice_vector(self: &mut Self, other: &FixedWidthSliceVector<T>) {
        assert_eq!(self.width, other.width);

        self.data.to_mut().extend_from_slice(&other.data);
    }

    pub fn load(buffer: &'a [u8], width: usize) -> Self {
        Self {
            data: Cow::from(unsafe { crate::io::load_bytes_as(&buffer[..]) }),
            width,
        }
    }

    pub fn read<I: Read>(mut reader: I, width: usize) -> Result<Self> {
        let mut buffer = Vec::new();
        buffer.resize(width * ::std::mem::size_of::<T>(), 0);

        let mut vec = Self::new(width);

        while let Ok(()) = reader.read_exact(&mut buffer) {
            vec.push(unsafe { crate::io::load_bytes_as(&buffer[..]) });
        }

        Ok(vec)
    }

    pub fn read_with_capacity<I: Read>(
        mut reader: I,
        width: usize,
        capacity: usize,
    ) -> Result<Self> {
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

        buffer.write_u64::<LittleEndian>(self.len() as u64)?;

        let zero_offset = Offset::try_from(0).unwrap();
        write(&[zero_offset], buffer)?;
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
                write(slice_buffer.as_slice(), buffer)?;
                offsets.push(Offset::try_from(total_count).unwrap());
            }
            value_pos = buffer.seek(SeekFrom::Current(0))?;

            buffer.seek(SeekFrom::Start(offset_pos))?;
            write(offsets.as_slice(), buffer)?;
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
        Self {
            data: Cow::Borrowed(&self.data),
            width: self.width,
        }
    }

    pub fn get<'b>(self: &'b Self, idx: usize) -> &'b [T]
    where
        'a: 'b,
    {
        let begin = idx * self.width;
        let end = (idx + 1) * self.width;

        &self.data[begin..end]
    }

    pub fn get_mut<'b>(self: &'b mut Self, idx: usize) -> &'b mut [T]
    where
        'a: 'b,
    {
        let begin = idx * self.width;
        let end = begin + self.width;

        &mut self.data.to_mut()[begin..end]
    }

    pub fn push(self: &mut Self, data: &[T]) {
        assert_eq!(self.width, data.len());

        self.data.to_mut().extend_from_slice(data);
    }

    pub fn len(self: &Self) -> usize {
        self.data.len() / self.width
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }

    pub fn width(self: &Self) -> usize {
        self.width
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        write(&self.data[..], buffer)?;
        let bytes_written = self.data.len() * std::mem::size_of::<T>();

        Ok(bytes_written)
    }
}

impl<'a, T: 'a + Clone + Send + Sync> FixedWidthSliceVector<'a, T> {
    pub fn par_iter<'b>(self: &'b Self) -> impl IndexedParallelIterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        self.data.par_chunks(self.width)
    }

    pub fn par_iter_mut<'b>(self: &'b mut Self) -> impl IndexedParallelIterator<Item = &'b mut [T]>
    where
        'a: 'b,
    {
        self.data.to_mut().par_chunks_mut(self.width)
    }
}

impl<'a, T: 'a + Clone, Offset: TryFrom<usize> + Copy> VariableWidthSliceVector<'a, T, Offset>
where
    usize: TryFrom<Offset>,
    <Offset as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    <usize as std::convert::TryFrom<Offset>>::Error: std::fmt::Debug,
{
    pub fn new() -> Self {
        Self {
            offsets: vec![Offset::try_from(0).unwrap()].into(),
            data: Vec::new().into(),
        }
    }

    pub fn extend_from_slice_vector(self: &mut Self, other: &VariableWidthSliceVector<T, Offset>) {
        let prev_len = usize::try_from(*self.offsets.last().unwrap()).unwrap();

        self.offsets.to_mut().extend(
            other
                .offsets
                .iter()
                .skip(1) // skip the initial 0
                .map(|&x| {
                    let x = usize::try_from(x).unwrap();
                    Offset::try_from(prev_len + x).unwrap()
                }),
        );

        self.data.to_mut().extend_from_slice(&other.data);
    }

    pub fn iter<'b>(self: &'b Self) -> impl Iterator<Item = &'b [T]>
    where
        'a: 'b,
    {
        self.offsets
            .iter()
            .zip(self.offsets.iter().skip(1))
            .map(move |(&begin, &end)| {
                let begin = usize::try_from(begin).unwrap();
                let end = usize::try_from(end).unwrap();

                &self.data[begin..end]
            })
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        let u64_len = ::std::mem::size_of::<u64>();
        let num_slices = (&buffer[..u64_len])
            .read_u64::<LittleEndian>()
            .expect("Could not read length") as usize;
        let offset_len = ::std::mem::size_of::<Offset>();
        let (offsets, data) = buffer[u64_len..].split_at((1 + num_slices) * offset_len);

        unsafe {
            Self {
                offsets: Cow::from(crate::io::load_bytes_as::<Offset>(offsets)),
                data: Cow::from(crate::io::load_bytes_as::<T>(data)),
            }
        }
    }

    pub fn borrow<'b>(self: &'a Self) -> VariableWidthSliceVector<'b, T, Offset>
    where
        'a: 'b,
    {
        Self {
            offsets: Cow::Borrowed(&self.offsets),
            data: Cow::Borrowed(&self.data),
        }
    }

    pub fn write_range<B: Write>(
        self: &Self,
        buffer: &mut B,
        begin: usize,
        end: usize,
    ) -> Result<()> {
        assert!(begin <= end);
        assert!(begin <= self.len());
        assert!(end <= self.len());

        // write metadata
        buffer
            .write_u64::<LittleEndian>((end - begin) as u64)
            .expect("Could not write length");

        let offset_begin = usize::try_from(self.offsets[begin]).unwrap();
        for i in begin..=end {
            let offset = usize::try_from(self.offsets[i]).unwrap();
            let offset = Offset::try_from(offset - offset_begin).unwrap();
            write(&[offset], buffer)?;
        }

        let data_begin = usize::try_from(self.offsets[begin]).unwrap();
        let data_end = usize::try_from(self.offsets[end]).unwrap();
        write(&self.data[data_begin..data_end], buffer)
    }

    pub fn get<'b>(self: &'b Self, idx: usize) -> &'b [T]
    where
        'a: 'b,
    {
        let begin = usize::try_from(self.offsets[idx]).unwrap();
        let end = usize::try_from(self.offsets[idx + 1]).unwrap();

        &self.data[begin..end]
    }

    pub fn get_mut<'b>(self: &'b mut Self, idx: usize) -> &'b mut [T]
    where
        'a: 'b,
    {
        let begin = usize::try_from(self.offsets[idx]).unwrap();
        let end = usize::try_from(self.offsets[idx + 1]).unwrap();

        &mut self.data.to_mut()[begin..end]
    }

    pub fn len(self: &Self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }

    pub fn push(self: &mut Self, data: &[T]) {
        self.data.to_mut().extend_from_slice(data);
        self.offsets
            .to_mut()
            .push(Offset::try_from(self.data.len()).unwrap());
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        // write metadata
        buffer
            .write_u64::<LittleEndian>(self.len() as u64)
            .expect("Could not write length");

        let mut bytes_written = std::mem::size_of::<u64>();

        write(&self.offsets[..], buffer)?;
        bytes_written += self.offsets.len() * std::mem::size_of::<Offset>();

        write(&self.data[..], buffer)?;
        bytes_written += self.data.len() * std::mem::size_of::<T>();

        Ok(bytes_written)
    }
}

fn write<T, B: Write>(elements: &[T], buffer: &mut B) -> Result<()> {
    let data = unsafe {
        ::std::slice::from_raw_parts(
            elements.as_ptr() as *const u8,
            elements.len() * ::std::mem::size_of::<T>(),
        )
    };

    buffer.write_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile;

    #[test]
    fn fixed_width_push() {
        let vec = FixedWidthSliceVector::new(5);
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
        let mut vec0 = FixedWidthSliceVector::new(7);
        let mut vec1 = FixedWidthSliceVector::new(7);
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
        let mut vec0 = FixedWidthSliceVector::new(7);
        let mut vec1 = FixedWidthSliceVector::new(6);
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

    fn test_extend_var(
        vec0: VariableWidthSliceVector<i32, usize>,
        vec1: VariableWidthSliceVector<i32, usize>,
    ) {
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
        let mut vec = FixedWidthSliceVector::new(width);
        for i in 0..123 {
            let data: Vec<i16> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut buffer = Vec::new();
        vec.write(&mut buffer).unwrap();

        let loaded_vec = FixedWidthSliceVector::<i16>::load(&buffer, width);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn fixed_width_write_and_read() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new(width);
        for i in 0..123 {
            let data: Vec<i16> = (2 * i + 3..).take(7).collect();
            vec.push(&data);
        }

        let mut buffer = Vec::new();
        vec.write(&mut buffer).unwrap();

        let read_vec = FixedWidthSliceVector::<i16>::read(&mut buffer.as_slice(), width).unwrap();

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

        let loaded_vec = VariableWidthSliceVector::<usize, usize>::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new(width);
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
            size_of::<usize>()
                + (1 + vec.len()) * size_of::<Offset>()
                + vec.len() * width * size_of::<i16>(),
            bytes_written
        );

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_predicate() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new(width);
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

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            let vec_slice: Vec<_> = vec
                .get(i)
                .iter()
                .filter(|&x| x % 3 == 0)
                .map(|x| *x)
                .collect();
            assert_eq!(vec_slice.as_slice(), loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_empty() {
        let width = 7;
        let vec = FixedWidthSliceVector::<i16>::new(width);

        type Offset = usize;

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_variable_width_slice_vector::<Offset, _, _>(&mut file, |_| true)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::load(&buffer);

        assert_eq!(0, loaded_vec.len());
    }

    #[test]
    fn write_fixed_width_vector_as_variable_width_vector_empty_slices() {
        let width = 1;
        let mut vec = FixedWidthSliceVector::<i16>::new(width);

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

        let loaded_vec = VariableWidthSliceVector::<i16, Offset>::load(&buffer);

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

                let loaded_vec = VariableWidthSliceVector::<usize, usize>::load(&buffer);

                assert_eq!(end - begin, loaded_vec.len());

                for i in begin..end {
                    assert_eq!(vec.get(i), loaded_vec.get(i - begin));
                }
            }
        }
    }
}
