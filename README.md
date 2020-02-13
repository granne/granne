granne\*
----

[![Crates.io](https://img.shields.io/crates/v/granne.svg)](https://crates.io/crates/granne)
[![documentation](https://docs.rs/granne/badge.svg)](https://docs.rs/granne)
[![license](http://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**granne** (**g**raph-based **r**etrieval of **a**pproximate **n**earest **ne**ighbors) is a Rust library for approximate nearest neighbor search based on [Hierarchical Navigable Small World (HNSW) graphs](https://arxiv.org/abs/1603.09320) and is used in [Cliqz Search](https://beta.cliqz.com). It focuses on reducing memory usage in order to allow [indexing billions of vectors](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html).

## Features
- Memory-mapped
- Multithreaded index creation
- Extensible indexes (add elements to an already built index)
- Python bindings
- Dense `float` or `int8` elements (cosine distance)

## Installation

#### Requirements

You will need to have `Rust` installed. This can be done by calling:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Or by visiting https://rustup.rs/ and following the instructions there.

#### Rust

```
# build
cargo build --release

# test
cargo test

# bench
cargo +nightly bench
```

#### Python

See [Python Bindings](py).

#### Optional Requirements

granne can use `BLAS` (https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) to improve speed of some computations. On Debian/Ubuntu both `libblas-dev` and `libopenblas-dev` should work, with the latter being significantly faster.

`BLAS` can be enabled by passing the `blas` feature during compilation, e.g.
```
cargo build --release --features "blas"
```

On Mac OS there seems to be some issue ([maybe
this one](https://grokbase.com/t/r/r-sig-mac/106pkkknqd/problems-with-single-precision-routines-in-64-bit-veclib-blas))
with the default `BLAS` library. A workaround is to install e.g. `openblas` and link to that instead.

----
\***granne** is Swedish and means **neighbor**
