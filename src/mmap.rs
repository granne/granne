use memmap::Mmap;
use At;
use Writeable;
use std::marker;
use file_io;
use std::fs::File;
use std::io::{Write, Result};
use madvise::{AccessPattern, AdviseMemory};

pub struct MmapSlice<Element> {
    path: String,
    data: Mmap,
    _marker: marker::PhantomData<Element>,
}

impl<Element> MmapSlice<Element>
{
    pub fn new(path: &str) -> Self {
        let data = File::open(path).unwrap();
        let data = unsafe { Mmap::map(&data).expect("Coud not read elements!") };
        data.advise_memory_access(AccessPattern::Random).expect("Error with madvise!");

        Self {
            path: path.to_string(),
            data: data,
            _marker: marker::PhantomData,
        }
    }

    pub fn as_slice(self: &Self) -> &[Element] {
        file_io::load::<Element>(&self.data[..])
    }
}

impl<Element: Clone> At for MmapSlice<Element> {
    type Output=Element;

    fn at(self: &Self, index: usize) -> Self::Output {
        self.as_slice()[index].clone()
    }

    fn len(self: &Self) -> usize {
        self.data.len() / ::std::mem::size_of::<Element>()
    }
}

impl<Element> Clone for MmapSlice<Element> {
    fn clone(self: &Self) -> Self {
        Self::new(&self.path)
    }
}

impl<T> Writeable for MmapSlice<T> {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        file_io::write(self.as_slice(), buffer)
    }
}
