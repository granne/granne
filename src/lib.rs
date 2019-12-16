#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod elements;
mod index;
mod io;
mod max_size_heap;
mod odd_byte_int;
mod slice_vector;
mod test_helper;

pub(crate) use odd_byte_int::{FiveByteInt, ThreeByteInt};

pub use elements::{angular, angular_int, sum_embeddings};
pub use elements::{Dist, ElementContainer, ExtendableElementContainer};
pub use index::{Config, Granne, GranneBuilder};

// only export these for benchmarks...
//#[cfg(test)]
#[doc(hidden)]
pub mod bench_helper {

    pub use crate::slice_vector::*;
    pub use crate::test_helper::*;
}
