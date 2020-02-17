//#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![deny(missing_docs)]

/*!
Granne (**g**raph-based **r**etrieval of **a**pproximate **n**earest **ne**ighbors) provides approximate nearest neighbor search among (typically) high-dimensional vectors. It focuses on reducing memory usage in order to allow [indexing billions of vectors](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html).

Note: There is currently a limit on the number of elements that can be indexed: `2^32 - 2 == 4_294_967_294`.

# Overview

* [`Granne`](struct.Granne.html) is the main struct used for querying/searching for nearest neighbors.
* [`GranneBuilder`](struct.GranneBuilder.html) is used to build `Granne` indexes.
* [`Index`](trait.Index.html) and [`Builder`](trait.Builder.html) are traits implemented for all `Granne` and `GranneBuilder` types, since both `Granne` and `GranneBuilder` are generic over the type of elements.
* [`SumEmbeddings`](embeddings/struct.SumEmbeddings.html) support the use case where the element are constructed by summing a smaller number of embeddings.

# Memory-mapping

Granne uses [memory-mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) for both index and elements. This makes it possible to lazy-load and use the index without loading all of it into memory (or share the index between different process).

Unfortunately, memory-mapping requires `unsafe` code to load the index/elements. The reason is that `Granne` cannot guarantee that the underlying file is not modified (by some other thread or process) while being memory-mapped.

For more information on the unsafe:ness of `memmap`, please see the [`memmap`](https://crates.io/crates/memmap) crate or refer to this discussion on
[users.rust-lang.org](https://users.rust-lang.org/t/how-unsafe-is-mmap/19635).

# Examples

## Basic building, saving and loading

For information on how to read/load/create elements see e.g. [`angular`](angular/index.html) or the `glove.rs` example.

```
use granne::{Granne, GranneBuilder, Index, Builder, BuildConfig, angular};

# use tempfile;
# const DIM: usize = 5;
# fn main() -> std::io::Result<()> {
# let elements: angular::Vectors = granne::test_helper::random_vectors(DIM, 1000);
# let random_vector: angular::Vector = granne::test_helper::random_vector(DIM);
# let num_results = 10;
# /*
let elements: angular::Vectors = /* omitted */
# */

// building the index
let mut builder = GranneBuilder::new(BuildConfig::default(), elements);
builder.build();

// saving to disk
let mut index_file = tempfile::tempfile()?;
builder.write_index(&mut index_file)?;

let mut elements_file = tempfile::tempfile()?;
builder.write_elements(&mut elements_file)?;

// loading (memory-mapping) index and vectors
let elements = unsafe { angular::Vectors::from_file(&elements_file)? };
let index = unsafe { Granne::from_file(&index_file, elements)? };

// max_search controls how extensive the search is
let max_search = 200;
let res = index.search(&random_vector, max_search, num_results);

assert_eq!(num_results, res.len());

# Ok(())
# }
```

*/

mod elements;
mod index;
mod io;
mod math;
mod max_size_heap;
mod odd_byte_int;
mod slice_vector;
use odd_byte_int::{FiveByteInt, ThreeByteInt};

pub use elements::{angular, angular_int, embeddings};
pub use elements::{Dist, ElementContainer, ExtendableElementContainer, Permutable};
pub use index::{BuildConfig, Builder, Granne, GranneBuilder, Index};
pub use io::Writeable;

#[cfg(feature = "rw_granne")]
pub use index::rw::RwGranneBuilder;

#[doc(hidden)]
pub mod test_helper;
