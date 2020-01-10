#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod elements;
mod index;
mod io;
mod math;
mod max_size_heap;
mod odd_byte_int;
use odd_byte_int::{FiveByteInt, ThreeByteInt};

pub use elements::{angular, angular_int, embeddings};
pub use elements::{Dist, ElementContainer, ExtendableElementContainer};
pub use index::{BuildConfig, Builder, Granne, GranneBuilder, Index};
pub use io::Writeable;

//#[doc(hidden)]
pub use index::RwGranneBuilder;

// only export these for benchmarks/doctests...
#[doc(hidden)]
pub mod test_helper;

#[doc(hidden)]
pub mod slice_vector;
