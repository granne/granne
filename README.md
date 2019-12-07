granne\*
--------
--------

**granne** (**g**raph-based **r**etrieval of **a**pproximate **n**earest **ne**ighbors) is a Rust library for ANN-search based on [Hierarchical Navigable Small World (HNSW) graphs](https://arxiv.org/abs/1603.09320) and is used in [Cliqz Search](https://beta.cliqz.com). For some background and motivation behind granne, please read [Indexing Billions of Text Vectors](https://0x65.dev/blog/2019-12-07/indexing-billions-of-text-vectors.html).

**Note: granne is still under active development. A more stable release (with documentation) is coming soon.**

## Features
- Memory-mapped
- Multithreaded index creation
- Extensible indexes (add elements to an already built index)
- Python bindings
- Dense `float` or `int8` elements (cosine distance)

## Installation

#### Requirements

`granne` is dependent on `BLAS` (https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) for some computations. This applies both to the rust and python versions. On Debian/Ubuntu both `libblas-dev` and `libopenblas-dev` should work, with the latter being significantly faster.

On Mac OS there seems to be some issue ([maybe
this one](https://grokbase.com/t/r/r-sig-mac/106pkkknqd/problems-with-single-precision-routines-in-64-bit-veclib-blas))
with the default `BLAS` library. A workaround is to install e.g. `openblas` and link to that instead.

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

To quickly install:

```
pip install setuptools_rust
pip install .
```

To build python wheels for python 2.7, 3.4, 3.5 and 3.6 (requires docker).
```
docker build -t granne_manylinux docker/manylinux/
docker run -v $(pwd):/granne/ granne_manylinux /opt/build_wheels.sh
```
The output is written to `wheels/` and can be installed by
```
pip install granne --no-index -f wheels/
```


## Index Creation
...

## Search
...

\***granne** is Swedish and means **neighbor**