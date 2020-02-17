use std::io::{Result, Write};

/// Loads a slice of bytes as a slice of type T
pub unsafe fn load_bytes_as<T>(bytes: &[u8]) -> &[T] {
    let elements: &[T] =
        ::std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / ::std::mem::size_of::<T>());

    elements
}

pub fn write_as_bytes<T, B: Write>(elements: &[T], buffer: &mut B) -> Result<usize> {
    let size = elements.len() * ::std::mem::size_of::<T>();
    let data = unsafe { ::std::slice::from_raw_parts(elements.as_ptr() as *const u8, size) };

    buffer.write_all(data)?;

    Ok(size)
}

/// A trait for types that are writeable to a buffer
pub trait Writeable {
    /// Writes `self` to `buffer`, if successful returns `Ok(num_bytes_written)`
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize>;
}
