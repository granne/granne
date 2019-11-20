use super::{write, FixedWidthSliceVector, VariableWidthSliceVector};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::convert::TryFrom;
use std::io::{Read, Result, Seek, SeekFrom, Write};
use stream_vbyte::{decode, encode, Scalar};

#[derive(Clone)]
pub struct MultiSetVectorT<'a, Offset>
where
    Offset: Copy + TryFrom<usize>,
    usize: TryFrom<Offset>,
{
    data: VariableWidthSliceVector<'a, u8, Offset>,
}

pub type MultiSetVector<'a> = MultiSetVectorT<'a, usize>;

const MIN_NUMBERS_TO_ENCODE: usize = 4;

impl<'a, Offset> MultiSetVectorT<'a, Offset>
where
    Offset: Copy + TryFrom<usize>,
    usize: TryFrom<Offset>,
    <Offset as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    <usize as std::convert::TryFrom<Offset>>::Error: std::fmt::Debug,
{
    pub fn new() -> Self {
        Self {
            data: VariableWidthSliceVector::new(),
        }
    }

    pub fn len(self: &Self) -> usize {
        self.data.len()
    }

    pub fn is_empty(self: &Self) -> bool {
        self.len() == 0
    }

    pub fn push(self: &mut Self, data: &[u32]) {
        let mut data = data.to_vec();

        data.sort();

        self.push_sorted(data);
    }

    pub fn push_sorted(self: &mut Self, data: Vec<u32>) {
        self.data.push(&set_encode(data));
    }

    pub fn extend_from_multi_set_vector(self: &mut Self, other: &Self) {
        self.data.extend_from_slice_vector(&other.data);
    }

    pub fn get(self: &Self, idx: usize) -> Vec<u32> {
        let encoded_data = self.data.get(idx);

        let count = encoded_data[0] as usize;
        let mut decoded_nums = Vec::new();
        decoded_nums.resize(std::cmp::max(MIN_NUMBERS_TO_ENCODE, count), 0);

        let bytes_decoded =
            decode::<Scalar>(&encoded_data[1..], decoded_nums.len(), &mut decoded_nums);

        debug_assert_eq!(encoded_data[1..].len(), bytes_decoded);

        decoded_nums.resize(count, 0);

        differential_decode(&mut decoded_nums);

        decoded_nums
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        self.data.write(buffer)
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        Self {
            data: VariableWidthSliceVector::load(buffer),
        }
    }
}

fn set_encode(mut data: Vec<u32>) -> Vec<u8> {
    debug_assert!(data.len() < u8::max_value() as usize);
    if data.len() >= u8::max_value() as usize {
        data.resize(u8::max_value() as usize, 0)
    }

    differential_encode(&mut data);
    let count = data.len() as u8;

    if data.len() < MIN_NUMBERS_TO_ENCODE {
        data.resize(MIN_NUMBERS_TO_ENCODE, 0);
    }

    let mut encoded_data = Vec::new();
    const MAX_REQUIRED_SIZE_PER_NUM: usize = 5;
    encoded_data.resize(MAX_REQUIRED_SIZE_PER_NUM * data.len(), 0x0);

    let encoded_len = encode::<Scalar>(data.as_slice(), &mut encoded_data);
    encoded_data.resize(encoded_len, 0x0);
    encoded_data.insert(0, count);

    encoded_data
}

fn differential_encode(data: &mut [u32]) {
    for i in (1..data.len()).rev() {
        data[i] -= data[i - 1];
    }
}

fn differential_decode(data: &mut [u32]) {
    for i in 1..data.len() {
        data[i] += data[i - 1];
    }
}

impl<'a, T: 'a + Clone> FixedWidthSliceVector<'a, T>
where
    u32: TryFrom<T>,
    <u32 as std::convert::TryFrom<T>>::Error: std::fmt::Debug,
{
    pub fn write_as_multi_set_vector<Offset, B, P>(
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

        let mut slice_buffer: Vec<u32> = Vec::new();
        let mut offsets: Vec<Offset> = Vec::new();

        // write to file in chunks (better for BufWriter)
        let mut num_chunks = 100;
        let chunk_size = std::cmp::max(100, self.len() / num_chunks);
        num_chunks = (self.len() + chunk_size - 1) / chunk_size;

        let mut total_len: usize = 0;
        for chunk in 0..num_chunks {
            // starting index for this chunk
            let chunk_offset = chunk * chunk_size;

            // chunk_size or whatever is left
            let chunk_size = std::cmp::min(chunk_size, self.len() - chunk_offset);

            offsets.clear();

            // write values
            buffer.seek(SeekFrom::Start(value_pos))?;
            for i in 0..chunk_size {
                slice_buffer.clear();
                for val in self.get(chunk_offset + i) {
                    if predicate(val) {
                        slice_buffer.push(u32::try_from(val.clone()).unwrap());
                    }
                }

                slice_buffer.sort();

                let encoded = set_encode(slice_buffer.clone());
                write(encoded.as_slice(), buffer)?;

                total_len += encoded.len();
                offsets.push(Offset::try_from(total_len).unwrap());
            }
            value_pos = buffer.seek(SeekFrom::Current(0))?;

            // write offsets
            buffer.seek(SeekFrom::Start(offset_pos))?;
            write(offsets.as_slice(), buffer)?;
            offset_pos = buffer.seek(SeekFrom::Current(0))?;
        }

        buffer.seek(SeekFrom::Start(value_pos))?;

        let bytes_written = (value_pos - initial_pos) as usize;

        Ok(bytes_written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile;

    #[test]
    fn test_differential_encode() {
        let mut data = vec![1, 2, 2, 4];

        differential_encode(&mut data);

        assert_eq!(vec![1, 1, 0, 2], data);
    }

    #[test]
    fn differential_encode_decode() {
        let data = vec![123, 345, 555, 555, 6999, 7000];
        let mut code = data.clone();

        differential_encode(&mut code);
        differential_decode(&mut code);

        assert_eq!(data, code);
    }

    #[test]
    fn push_and_get() {
        let mut vec = MultiSetVector::new();

        let slice: Vec<u32> = (0..10).collect();

        vec.push(&slice);

        assert_eq!(1, vec.len());
        assert_eq!(slice, vec.get(0));
    }

    #[test]
    fn push_and_get_empty() {
        let mut vec = MultiSetVector::new();

        vec.push(&[]);

        assert_eq!(1, vec.len());
        assert_eq!(Vec::<u32>::new(), vec.get(0));
    }

    #[test]
    fn push_and_get_one() {
        let mut vec = MultiSetVector::new();

        vec.push(&[5]);

        assert_eq!(1, vec.len());
        assert_eq!(vec![5], vec.get(0));
    }

    #[test]
    fn push_and_get_duplicates() {
        let mut vec = MultiSetVector::new();

        vec.push(&[5, 5]);

        assert_eq!(1, vec.len());
        assert_eq!(vec![5, 5], vec.get(0));
    }

    #[test]
    fn push_multiple_and_get() {
        let mut vec = MultiSetVector::new();

        for i in 0..20 {
            let slice: Vec<u32> = (i..20).collect();

            vec.push(&slice);

            assert_eq!((i + 1) as usize, vec.len());
            assert_eq!(slice, vec.get(i as usize));
        }
    }

    #[test]
    fn push_unsorted() {
        let mut vec = MultiSetVector::new();

        for i in 0..20 {
            let mut slice: Vec<u32> = (i..20).map(|i| (7 * i) % 15).collect();

            vec.push(&slice);
            slice.sort();

            assert_eq!((i + 1) as usize, vec.len());
            assert_eq!(slice, vec.get(i as usize));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..123 {
            let data: Vec<u32> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut file: File = tempfile::tempfile().unwrap();
        let bytes_written = vec
            .write_as_multi_set_vector::<usize, _, _>(&mut file, |_| true)
            .unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            assert_eq!(vec.get(i), loaded_vec.get(i).as_slice());
        }
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector_predicate() {
        let width = 7;
        let mut vec = FixedWidthSliceVector::new();
        for i in 0..522 {
            let data: Vec<u32> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_multi_set_vector::<usize, _, _>(&mut file, |x| x % 3 == 0)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            let vec_slice: Vec<_> = vec
                .get(i)
                .iter()
                .filter(|&x| x % 3 == 0)
                .map(|x| *x)
                .collect();
            assert_eq!(vec_slice, loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector_empty() {
        let width = 7;
        let vec = FixedWidthSliceVector::<u16>::new();

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_multi_set_vector::<usize, _, _>(&mut file, |_| true)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::load(&buffer);

        assert_eq!(0, loaded_vec.len());
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector_empty_slices() {
        let width = 1;
        let mut vec = FixedWidthSliceVector::<u32>::new();

        for _ in 0..10 {
            vec.push(&[0]);
        }

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_multi_set_vector::<usize, _, _>(&mut file, |&x| x > 0)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::load(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..loaded_vec.len() {
            assert!(loaded_vec.get(i).is_empty());
        }
    }
}
