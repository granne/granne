use crate::{SliceVector, VariableWidthSliceVector};
use super::write;
use std::io::{Result, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::borrow::Cow;
use stream_vbyte::{decode, encode, Scalar};

#[derive(Clone)]
pub struct MultiSetVector<'a> {
    data: VariableWidthSliceVector<'a, u8, usize>,
    counts: Cow<'a, [u8]>,
}

const MIN_NUMBERS_TO_ENCODE: usize = 4;

impl<'a> MultiSetVector<'a> {
    pub fn new() -> Self {
        Self {
            data: VariableWidthSliceVector::new(),
            counts: Vec::new().into(),
        }
    }

    pub fn len(self: &Self) -> usize {
        self.data.len()
    }

    pub fn push(self: &mut Self, data: &[u32]) {
        let mut data = data.to_vec();

        data.sort();

        self.push_sorted(data);
    }

    pub fn push_sorted(self: &mut Self, mut data: Vec<u32>) {
        debug_assert!(data.len() < <u8>::max_value() as usize);
        if data.len() >= <u8>::max_value() as usize {
            data.resize(<u8>::max_value() as usize, 0)
        }

        Self::differential_encode(&mut data);

        self.counts.to_mut().push(data.len() as u8);

        if data.len() < MIN_NUMBERS_TO_ENCODE {
            data.resize(MIN_NUMBERS_TO_ENCODE, 0);
        }

        let mut encoded_data = Vec::new();
        const MAX_REQUIRED_SIZE_PER_NUM: usize = 5;
        encoded_data.resize(MAX_REQUIRED_SIZE_PER_NUM * data.len(), 0x0);

        let encoded_len = encode::<Scalar>(data.as_slice(), &mut encoded_data);

        self.data.push(&encoded_data[..encoded_len]);
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

    pub fn extend_from_multi_set_vector(self: &mut Self, other: &MultiSetVector) {
        self.data.extend_from_slice_vector(&other.data);
        self.counts.to_mut().extend_from_slice(&other.counts);
    }

    pub fn get(self: &Self, idx: usize) -> Vec<u32> {
        let count = std::cmp::max(MIN_NUMBERS_TO_ENCODE, self.counts[idx] as usize);

        let mut decoded_nums = Vec::new();
        decoded_nums.resize(count, 0);

        let encoded_data = self.data.get(idx);

        let bytes_decoded = decode::<Scalar>(encoded_data, count, &mut decoded_nums);

        assert_eq!(encoded_data.len(), bytes_decoded);

        if (self.counts[idx] as usize) < MIN_NUMBERS_TO_ENCODE {
            decoded_nums.resize(self.counts[idx] as usize, 0);
        }

        Self::differential_decode(&mut decoded_nums);

        decoded_nums
    }

    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        buffer
            .write_u64::<LittleEndian>(self.len() as u64)
            .expect("Could not write length");

        write(&self.counts[..], buffer)?;
        self.data.write(buffer)
    }

    pub fn load(buffer: &'a [u8]) -> Self {
        let u64_len = ::std::mem::size_of::<u64>();
        let num_sets = (&buffer[..u64_len])
            .read_u64::<LittleEndian>()
            .expect("Could not read length") as usize;
        let (counts, data) = buffer[u64_len..].split_at(num_sets);

        Self {
            counts: Cow::from(counts),
            data: VariableWidthSliceVector::load(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differential_encode() {
        let mut data = vec![1, 2, 2, 4];

        MultiSetVector::differential_encode(&mut data);

        assert_eq!(vec![1, 1, 0, 2], data);
    }

    #[test]
    fn differential_encode_decode() {
        let mut data = vec![123, 345, 555, 555, 6999, 7000];
        let mut code = data.clone();

        MultiSetVector::differential_encode(&mut code);
        MultiSetVector::differential_decode(&mut code);

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
}
