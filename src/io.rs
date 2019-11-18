use std::io::{Result, Write};

/// Loads a slice of bytes as a slice of type T
pub unsafe fn load_bytes_as<T>(bytes: &[u8]) -> &[T] {
    let elements: &[T] = ::std::slice::from_raw_parts(
        bytes.as_ptr() as *const T,
        bytes.len() / ::std::mem::size_of::<T>(),
    );

    elements
}

/// Loads a slice of a type T as a slice of bytes
pub fn load_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        ::std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * ::std::mem::size_of::<T>(),
        )
    }
}

/// A trait for types that are writeable to a buffer
pub trait Writeable {
    /// Writes `self` to `buffer`, if successful returns `Ok(num_bytes_written)`
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<usize>;
}
