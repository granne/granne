use super::{CompressedVariableWidthSliceVector, FixedWidthSliceVector, VariableWidthSliceVector};
use crate::io::write_as_bytes;
use std::convert::TryFrom;
use std::io::{Read, Result, Seek, SeekFrom, Write};
use stream_vbyte::{decode, encode, Scalar};

#[derive(Clone)]
pub struct MultiSetVector<'a> {
    data: CompressedVariableWidthSliceVector<'a, u8>,
}

const MIN_NUMBERS_TO_ENCODE: usize = 4;

impl<'a> MultiSetVector<'a> {
    pub fn new() -> Self {
        Self {
            data: CompressedVariableWidthSliceVector::new(),
        }
    }

    pub unsafe fn from_file(file: &std::fs::File) -> std::io::Result<Self> {
        Ok(Self {
            data: CompressedVariableWidthSliceVector::from_file(file)?,
        })
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        Self {
            data: CompressedVariableWidthSliceVector::from_bytes(buffer),
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
    /*
        pub fn extend_from_multi_set_vector(self: &mut Self, other: &Self) {
            self.data.extend_from_slice_vector(&other.data);
        }
    */
    pub fn get(self: &Self, idx: usize) -> Vec<u32> {
        let mut decoded_nums = Vec::new();

        self.get_into(idx, &mut decoded_nums);

        decoded_nums
    }

    pub fn get_into(self: &Self, idx: usize, res: &mut Vec<u32>) {
        let encoded_data = self.data.get(idx);

        decode_into(&encoded_data, res);
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize> {
        self.data.write(buffer)
    }

    pub fn borrow<'b>(self: &'a Self) -> MultiSetVector<'b>
    where
        'a: 'b,
    {
        Self {
            data: self.data.borrow(),
        }
    }

    pub fn into_owned(self: Self) -> MultiSetVector<'static> {
        MultiSetVector {
            data: self.data.into_owned(),
        }
    }
}

fn decode_into(encoded_data: &[u8], decoded_nums: &mut Vec<u32>) {
    let count = encoded_data[0] as usize;
    decoded_nums.clear();
    let encoded_data = &encoded_data[1..];

    if encoded_data.len() != count * ::std::mem::size_of::<u32>() {
        decoded_nums.resize(std::cmp::max(MIN_NUMBERS_TO_ENCODE, count), 0);

        //let bytes_decoded = decode::<stream_vbyte::x86::Ssse3>(encoded_data, decoded_nums.len(),
        // decoded_nums);
        let bytes_decoded = decode::<Scalar>(encoded_data, decoded_nums.len(), decoded_nums);

        debug_assert_eq!(encoded_data.len(), bytes_decoded);

        decoded_nums.resize(count, 0);
    } else {
        let mut buf = [0x0; ::std::mem::size_of::<u32>()];
        for num in encoded_data.chunks_exact(::std::mem::size_of::<u32>()).take(count) {
            buf.copy_from_slice(num);
            decoded_nums.push(u32::from_le_bytes(buf));
        }
    }

    delta_decode(decoded_nums);
}

fn set_encode(mut data: Vec<u32>) -> Vec<u8> {
    debug_assert!(data.len() < u8::max_value() as usize);
    if data.len() >= u8::max_value() as usize {
        data.resize(u8::max_value() as usize, 0)
    }

    delta_encode(&mut data);
    let count = data.len();

    if data.len() < MIN_NUMBERS_TO_ENCODE {
        data.resize(MIN_NUMBERS_TO_ENCODE, 0);
    }

    let mut encoded_data = Vec::new();
    const MAX_REQUIRED_SIZE_PER_NUM: usize = 5;
    encoded_data.resize(MAX_REQUIRED_SIZE_PER_NUM * data.len(), 0x0);

    let encoded_len = encode::<Scalar>(data.as_slice(), encoded_data.as_mut_slice());
    encoded_data.resize(encoded_len, 0x0);

    // only use compression if it makes the data smaller
    if encoded_data.len() >= ::std::mem::size_of::<u32>() * count {
        encoded_data.clear();
        for d in data.iter().take(count) {
            encoded_data.extend_from_slice(&d.to_le_bytes());
        }
    }

    encoded_data.insert(0, count as u8);

    encoded_data
}

#[inline(always)]
fn delta_encode(data: &mut [u32]) {
    for i in (1..data.len()).rev() {
        data[i] -= data[i - 1];
    }
}

#[inline(always)]
fn delta_decode(data: &mut [u32]) {
    for i in 1..data.len() {
        data[i] += data[i - 1];
    }
}

impl<'a, T: 'a + Clone> FixedWidthSliceVector<'a, T>
where
    u32: TryFrom<T>,
    <u32 as std::convert::TryFrom<T>>::Error: std::fmt::Debug,
{
    pub fn write_as_multi_set_vector<B, P>(self: &Self, buffer: &mut B, mut predicate: P) -> Result<usize>
    where
        B: Write + Seek,
        P: FnMut(&T) -> bool,
    {
        let initial_pos = buffer.seek(SeekFrom::Current(0))?;

        let bytes_for_offsets =
            (1 + self.len() / super::offsets::OFFSETS_PER_CHUNK) * ::std::mem::size_of::<super::offsets::Chunk>();

        buffer.write_all(&(bytes_for_offsets as u64).to_le_bytes())?;

        let mut offset_pos = buffer.seek(SeekFrom::Current(0))?;
        let mut offsets = super::Offsets::new();
        offsets.push(0);

        // write values
        buffer.seek(SeekFrom::Start(offset_pos + bytes_for_offsets as u64))?;
        let mut slice_buffer: Vec<u32> = Vec::new();
        let mut total_len: usize = 0;
        if !self.is_empty() {
            for slice in self.iter() {
                slice_buffer.clear();
                for val in slice {
                    if predicate(val) {
                        slice_buffer.push(u32::try_from(val.clone()).unwrap());
                    }
                }

                slice_buffer.sort();

                let encoded = set_encode(slice_buffer.clone());
                write_as_bytes(encoded.as_slice(), buffer)?;

                total_len += encoded.len();
                offsets.push(total_len);
            }
        }

        let value_pos = buffer.seek(SeekFrom::Current(0))?;

        // write offsets
        buffer.seek(SeekFrom::Start(offset_pos))?;
        let offsets_bytes_written = offsets.write(buffer)?;

        assert_eq!(offsets_bytes_written, bytes_for_offsets);

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
    fn test_delta_encode() {
        let mut data = vec![1, 2, 2, 4];

        delta_encode(&mut data);

        assert_eq!(vec![1, 1, 0, 2], data);
    }

    #[test]
    fn delta_encode_decode() {
        let data = vec![123, 345, 555, 555, 6999, 7000];
        let mut code = data.clone();

        delta_encode(&mut code);
        delta_decode(&mut code);

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
    // this tests the case where the size of the encoded data has the same size as the original (2 *
    // 4 bytes)
    fn push_and_get_4_bytes_per_number() {
        let mut vec = MultiSetVector::new();

        let exp = vec![37717, 660380];
        vec.push(&exp);

        assert_eq!(1, vec.len());
        assert_eq!(exp, vec.get(0));
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
        for i in 0..120 {
            let data: Vec<u32> = (2 * i + 3..).take(width).collect();
            vec.push(&data);
        }

        let mut file: File = tempfile::tempfile().unwrap();
        let bytes_written = vec.write_as_multi_set_vector(&mut file, |_| true).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::from_bytes(&buffer);

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
        vec.write_as_multi_set_vector(&mut file, |x| x % 3 == 0).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..vec.len() {
            let vec_slice: Vec<_> = vec.get(i).iter().filter(|&x| x % 3 == 0).map(|x| *x).collect();
            assert_eq!(vec_slice, loaded_vec.get(i));
        }
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector_empty() {
        let vec = FixedWidthSliceVector::<u16>::new();

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_multi_set_vector(&mut file, |_| true).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::from_bytes(&buffer);

        assert_eq!(0, loaded_vec.len());
    }

    #[test]
    fn write_fixed_width_vector_as_multi_set_vector_empty_slices() {
        let width = 1;
        let mut vec = FixedWidthSliceVector::<u32>::new();

        for _ in 0..1000 {
            vec.push(&[0]);
        }

        let mut file: File = tempfile::tempfile().unwrap();
        vec.write_as_multi_set_vector(&mut file, |&x| x > 0).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let loaded_vec = MultiSetVector::from_bytes(&buffer);

        assert_eq!(vec.len(), loaded_vec.len());

        for i in 0..loaded_vec.len() {
            assert!(loaded_vec.get(i).is_empty());
        }
    }
}
