#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod elements;
mod index;
mod io;
mod max_size_heap;
mod odd_byte_int;
use odd_byte_int::{FiveByteInt, ThreeByteInt};

pub use elements::{angular, angular_int, embeddings};
pub use elements::{Dist, ElementContainer, ExtendableElementContainer};
pub use index::{Granne, GranneBuilder};
pub use io::Writeable;

// only export these for benchmarks/doctests...
#[doc(hidden)]
pub mod test_helper;

#[doc(hidden)]
pub mod slice_vector;
