#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod index;
mod io;
mod max_size_heap;
mod odd_byte_int;
mod slice_vector;

pub(crate) use odd_byte_int::{FiveByteInt, ThreeByteInt};

mod elements;
pub use elements::{
    AngularIntVector, AngularIntVectors, AngularVector, AngularVectorT, AngularVectors,
    AngularVectorsT, Dist, ElementContainer, ExtendableElementContainer, SumEmbeddings,
};
pub use index::{Config, Granne, GranneBuilder};

// only export these for benchmarks...
//#[cfg(test)]
pub mod test_helper;
pub use slice_vector::*;
