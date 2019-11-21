#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod index;
mod io;
mod max_size_heap;
mod slice_vector;

pub mod elements;
pub use elements::{AngularIntVector, AngularIntVectors, AngularVector, AngularVectors, Dist};
pub use index::{Config, Granne, GranneBuilder};

//#[cfg(test)]
pub mod test_helper;
