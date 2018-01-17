granne\*
--------
--------

**granne** (**g**raph-based **r**etrieval of **a**pproximate **n**earest **ne**ighbors) is a Rust library for ANN-search based on Hierarchical Navigable Small World (HNSW) graphs (see https://arxiv.org/abs/1603.09320).

## Features
- Memory-mapped
- Multithreaded index creation
- Extensible indexes (add elements to an already built index)
- Python bindings
- Dense `float` or `int8` elements (cosine distance)

## Installation

The `rblas` crate has `blas` (e.g. `libopenblas-dev` on Debian/Ubuntu) as dependency. See https://www.crates.io/crates/rblas for details.

#### Rust

```
# build
cargo build

# test
cargo test

# bench (requires nightly rust)
cargo bench
```

#### Python

```
pip install setuptools_rust
pip install .
```

## Index Creation
...

## Search
...

\***granne** is Swedish and means **neighbor**