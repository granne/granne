use crate::file_io;
use crate::At;
use crate::Writeable;

use super::angular_vector::AngularVectorT;

use madvise::{AccessPattern, AdviseMemory};
use memmap::Mmap;
use std::fs::File;
use std::io::{Result, Write};
use std::marker;

pub struct MmapAngularVectorsT<T: Copy + 'static> {
    path: String,
    data: Mmap,
    dim: usize,
    _marker: marker::PhantomData<T>,
}

pub type MmapAngularVectors = MmapAngularVectorsT<f32>;
pub type MmapAngularIntVectors = MmapAngularVectorsT<i8>;

impl<T: Copy> MmapAngularVectorsT<T> {
    pub fn new(path: &str, dim: usize) -> Self {
        let data = File::open(path).unwrap();
        let data = unsafe { Mmap::map(&data).expect("Coud not read elements!") };
        data.advise_memory_access(AccessPattern::Random)
            .expect("Error with madvise!");

        Self {
            path: path.to_string(),
            data: data,
            dim: dim,
            _marker: marker::PhantomData,
        }
    }

    pub fn as_slice(self: &Self) -> &[T] {
        file_io::load::<T>(&self.data[..])
    }
}

impl<T: Copy + 'static> At for MmapAngularVectorsT<T> {
    type Output = AngularVectorT<'static, T>;

    fn at(self: &Self, index: usize) -> Self::Output {
        AngularVectorT(
            self.as_slice()[index * self.dim..(index + 1) * self.dim]
                .to_vec()
                .into(),
        )
    }

    fn len(self: &Self) -> usize {
        self.as_slice().len() / self.dim
    }
}

impl<T: Copy> Clone for MmapAngularVectorsT<T> {
    fn clone(self: &Self) -> Self {
        Self::new(&self.path, self.dim)
    }
}

impl<T: Copy> Writeable for MmapAngularVectorsT<T> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        file_io::write(self.as_slice(), buffer)
    }
}
