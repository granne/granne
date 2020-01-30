#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
//#![deny(missing_docs)]

/*!
Granne (**gr**aph-based **a**pproximate **n**earest **ne**ighbors) provides
approximate nearest neighbor search among (typically) high-dimensional vectors.

Supports billion scale indexes

Discussion on RAM vs. SSD, during build and search time.

Note: There is currently a limit on the number of elements that can be
indexed: `2^32 - 2 == 4_294_967_294`.

# Overview

* [`Granne`](struct.Granne.html) what is this?
* [`GranneBuilder`](struct.GranneBuilder.html) what is this?
* [`Index`](trait.Index.html) and [`Builder`](trait.Builder.html) are traits that ... Since both `Granne`
and `GranneBuilder` are generic over the type of elements, these traits contain methods that are the same
over all element types.

# Memory-mapping

Granne allows memory-mapping both index and elements. ---. Unfortunately, xxx

Using the [`memmap`](https://crates.io/crates/memmap) crate.

For more information on the unsafe:ness of `memmap`, please refer to this discussion on
[users.rust-lang.org](https://users.rust-lang.org/t/how-unsafe-is-mmap/19635).

# Summed Embeddings

An important use case is ...

# Examples

## Basic building, saving and loading

For information on how to read/load/create elements see e.g. [`angular`](angular/index.html).

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

## Reordering

Reordering of the index can be useful in a couple of situation:
* Since the neighbors of each node in the index is stored using a variable int encoding, reordering might make size of index smaller.
* Improve data locality for serving the index from disk. See
[Indexing Billions of Text Vectors](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html)
 for a more thorough explanation of this use case.

```
# use granne::{Granne, GranneBuilder, Index, Builder, BuildConfig, angular};
# use tempfile;
# const DIM: usize = 5;
# fn main() -> std::io::Result<()> {
# let elements: angular::Vectors = granne::test_helper::random_vectors(DIM, 1000);
# let random_vector: angular::Vector = granne::test_helper::random_vector(DIM);
# let num_results = 10;
#
# let mut builder = GranneBuilder::new(BuildConfig::default(), elements);
# builder.build();
# let mut index_file = tempfile::tempfile()?;
# builder.write_index(&mut index_file)?;
# let mut elements_file = tempfile::tempfile()?;
# builder.write_elements(&mut elements_file)?;
# let max_search = 10;
# let num_neighbors = 10;
use granne::{angular, Granne};

// loading index and vectors (original)
let elements = unsafe { angular::Vectors::from_file(&elements_file)? };
let index = unsafe { Granne::from_file(&index_file, elements)? };

// loading index and vectors (for reordering)
let elements = unsafe { angular::Vectors::from_file(&elements_file)? };
let mut reordered_index = unsafe { Granne::from_file(&index_file, elements)? };
let order = reordered_index.reorder(false);

// verify that results are the same
let element = index.get_element(123);
let res = index.search(&element, max_search, num_neighbors);
let reordered_res = reordered_index.search(&element, max_search, num_neighbors);
for (r, rr) in res.iter().zip(&reordered_res) {
    assert_eq!(r.0, order[rr.0]);
}

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
use odd_byte_int::{FiveByteInt, ThreeByteInt};

pub use elements::{angular, angular_int, embeddings};
pub use elements::{Dist, ElementContainer, ExtendableElementContainer, Permutable};
pub use index::{BuildConfig, Builder, Granne, GranneBuilder, Index};
pub use io::Writeable;

#[cfg(feature = "rw_granne")]
pub use index::RwGranneBuilder;

// only export these for benchmarks/doctests...
#[doc(hidden)]
pub mod test_helper;

#[doc(hidden)]
pub mod slice_vector;
