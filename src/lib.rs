#![warn(clippy::all, clippy::pedantic, clippy::cargo)]

mod granne;
mod io;
mod max_size_heap;
mod slice_vector;

pub mod elements;
pub use elements::{AngularIntVector, AngularIntVectors, AngularVector, AngularVectors};
pub use granne::{Granne, GranneBuilder};
