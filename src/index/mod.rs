use fxhash::FxBuildHasher;
use ordered_float::NotNan;
use parking_lot;
use pbr;
use rayon::prelude::*;
use revord::RevOrd;
use std::cmp;
use std::collections::{BinaryHeap, HashSet};
use std::convert::TryFrom;
use time;

#[cfg(test)]
mod tests;

mod io;
mod reorder;
mod rw;

pub use rw::RwGranneBuilder;

use crate::{
    max_size_heap,
    slice_vector::{FixedWidthSliceVector, MultiSetVector, VariableWidthSliceVector},
    {ElementContainer, ExtendableElementContainer, Permutable},
};

type NeighborId = u32;
const UNUSED: NeighborId = NeighborId::max_value();

/// An index for fast approximate nearest neighbor search.
/// The index is built by using `GranneBuilder` and can be stored to disk.
pub struct Granne<'a, Elements: ElementContainer> {
    layers: Layers<'a>,
    elements: Elements,
}

pub trait Index {
    fn len(self: &Self) -> usize;
    fn num_layers(self: &Self) -> usize;
    fn layer_len(self: &Self, layer: usize) -> usize;
    fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize>;
}

impl<'a, Elements: ElementContainer> Index for Granne<'a, Elements> {
    /// Returns the number of elements in this index.
    /// Note that it might be less than the number of elements in `elements`.
    fn len(self: &Self) -> usize {
        if self.layers.len() > 0 {
            self.layer_len(self.layers.len() - 1)
        } else {
            0
        }
    }

    /// Returns the number of layers in this index.
    fn num_layers(self: &Self) -> usize {
        self.layers.len()
    }

    /// Returns the number of nodes in `layer`.
    fn layer_len(self: &Self, layer: usize) -> usize {
        self.layers.as_graph(layer).len()
    }

    /// Returns the neighbors of the node at `index` in `layer`.
    fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize> {
        self.layers.as_graph(layer).get_neighbors(index)
    }
}

impl<'a, Elements: ElementContainer> Granne<'a, Elements> {
    /// Loads this index lazily.
    pub fn load(index: &'a [u8], elements: Elements) -> Self {
        Self {
            layers: io::load_layers(index),
            elements,
        }
    }

    /// Searches for the `num_neighbors` neighbors closest to `element` in this index.
    /// `max_search` controls the number of nodes visited during the search. Returns a
    /// `Vec` containing the id and distance from `element`.
    pub fn search(
        self: &Self,
        element: &Elements::Element,
        max_search: usize,
        num_neighbors: usize,
    ) -> Vec<(usize, f32)> {
        match &self.layers {
            Layers::FixWidth(layers) => {
                self.search_internal(&layers, element, max_search, num_neighbors)
            }
            Layers::Compressed(layers) => {
                self.search_internal(&layers, element, max_search, num_neighbors)
            }
        }
    }

    /// Returns the element at `index`.
    pub fn get_element(self: &Self, index: usize) -> Elements::Element {
        self.elements.get(index)
    }
}

/// Returns a vector with the id of the "closest" element in each layer.
fn find_entrypoint_trail<Elements: ElementContainer>(
    layers: &Layers,
    elements: &Elements,
    max_layer: usize,
    element: &Elements::Element,
) -> Vec<usize> {
    fn _find_entrypoint_trail<Layer: Graph, Elements: ElementContainer>(
        layers: &[Layer],
        elements: &Elements,
        max_layer: usize,
        element: &Elements::Element,
    ) -> Vec<usize> {
        let mut eps = Vec::new();
        eps.reserve(layers.len());
        for (i, layer) in layers.iter().enumerate().take(max_layer) {
            let ep = *eps.last().unwrap_or(&0);
            let max_search = if layer.len() < 10_000 { 5 } else { 10 };
            let res = search_for_neighbors(layer, ep, elements, element, max_search);
            eps.push(res[0].0);
        }
        eps
    }

    match layers {
        Layers::FixWidth(layers) => _find_entrypoint_trail(layers, elements, max_layer, element),
        Layers::Compressed(layers) => _find_entrypoint_trail(layers, elements, max_layer, element),
    }
}

impl<'a, Elements: ElementContainer + Permutable + crate::io::Writeable + Sync>
    Granne<'a, Elements>
{
    /// Reorders the elements in this index. Returns the permutation used for the
    /// reordering. `permutation[i] == j`, means that the element with idx `j`, has
    /// been moved to idx `i`.
    pub fn reorder(self: &mut Self, show_progress: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.len()).collect();
        let mut order_inv = vec![0; self.layer_len(self.num_layers() - 2)];

        let start_time = time::PreciseTime::now();
        let progress_bar = if show_progress {
            println!("Computing ordering...");

            Some(parking_lot::Mutex::new(pbr::ProgressBar::new(
                self.len() as u64
            )))
        } else {
            None
        };

        let step_size = cmp::max(100, self.len() / 400);
        for layer in 1..self.num_layers() {
            let eps: Vec<_> = (self.layer_len(layer - 1)..self.layer_len(layer))
                .into_par_iter()
                .map(|idx| {
                    if idx % step_size == 0 {
                        progress_bar.as_ref().map(|pb| pb.lock().add(step_size as u64));
                    }

                    let mut eps = find_entrypoint_trail(
                        &self.layers,
                        &self.elements,
                        layer,
                        &self.get_element(idx),
                    );
                    eps.iter_mut().for_each(|i| *i = order_inv[*i]);

                    eps
                })
                .collect();

            order[self.layer_len(layer - 1)..self.layer_len(layer)]
                .par_sort_by_key(|idx| &eps[idx - self.layer_len(layer - 1)]);

            if layer < self.num_layers() - 1 {
                for i in self.layer_len(layer - 1)..self.layer_len(layer) {
                    order_inv[order[i]] = i;
                }
            }
        }

        progress_bar.map(|pb| pb.lock().set(self.len() as u64));

        if show_progress {
            println!("Ordering computed!");
            println!("Reordering index...");
        }

        self.layers = reorder::reorder_layers(&self.layers, &order, show_progress);

        if show_progress {
            println!("Reordering elements...");
        }

        self.elements.permute(&order);

        if show_progress {
            println!(
                "Total time: {} s",
                start_time.to(time::PreciseTime::now()).num_seconds()
            );
        }

        order
    }

    pub fn reorder_and_save(
        self: &mut Self,
        index: &mut std::fs::File,
        elements: &mut std::fs::File,
        show_progress: bool,
    ) -> std::io::Result<Vec<usize>> {
        let order = self.reorder(show_progress);

        io::write_index(&self.layers, index)?;
        self.elements.write(elements)?;

        Ok(order)
    }
}

/// `BuildConfig` is used to configure a `GranneBuilder`.
///
/// # Examples
/// ```
/// # use granne::*;
/// let config = BuildConfig::new()
///     .num_neighbors(30)
///     .layer_multiplier(15.0)
///     .max_search(200);
/// let mut builder = GranneBuilder::new(config, angular::Vectors::new());
/// ```
#[derive(Copy, Clone, Debug)]
pub struct BuildConfig {
    /// Each layer includes `layer_multiplier` times more elements than the previous layer.
    layer_multiplier: f32,

    /// Needs to be used when building before all elements have been inserted into the builder.
    expected_num_elements: Option<usize>,

    /// The maximum number of neighbors per node and layer.
    num_neighbors: usize,

    /// The `max_search` parameter used during build time (see `granne::search`).
    max_search: usize,

    /// Whether to reinsert all the elements in each layers. Takes more time, but improves recall.
    reinsert_elements: bool,

    /// Whether to output progress information to STDOUT while building.
    show_progress: bool,
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            layer_multiplier: 15.0,
            expected_num_elements: None,
            num_neighbors: 30,
            max_search: 200,
            reinsert_elements: true,
            show_progress: false,
        }
    }
}

impl BuildConfig {
    /// Creates a BuildConfig for GranneBuilder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures the maximum number of neighbors per node in each layer.
    ///
    /// Default: 30
    pub fn num_neighbors(mut self: Self, num_neighbors: usize) -> Self {
        self.num_neighbors = num_neighbors;
        self
    }

    /// Configures the `max_search` parameter used during build time (see `Granne::search`).
    /// Larger values increase recall but makes building slower.
    ///
    /// Default: 200
    pub fn max_search(mut self: Self, max_search: usize) -> Self {
        self.max_search = max_search;
        self
    }

    /// Configures the expected number of elements in the final graph. This is only required if
    /// when building (`builder.build()`) before all elements have been inserted into the builder.
    pub fn expected_num_elements(mut self: Self, expected_num_elements: usize) -> Self {
        self.expected_num_elements = Some(expected_num_elements);
        self
    }

    /// Configures the layer multiplier for the hierarchical graph:
    /// each new layer will have `layer_multiplier` times more elements than the previous.
    /// E.g. `layer_multiplier == 10.0` implies `n`, `10n`, `100n`, ... nodes per layer for
    /// some `n < 10`.
    ///
    /// Default: 15.0
    pub fn layer_multiplier(mut self: Self, layer_multiplier: f32) -> Self {
        self.layer_multiplier = layer_multiplier;
        self
    }

    /// Enables reinsertion of all the elements in each layers. Takes more time, but improves recall.
    ///
    /// This option is enabled by default.
    pub fn reinsert_elements(mut self: Self, yes: bool) -> Self {
        self.reinsert_elements = yes;
        self
    }

    /// Enables printing progress information to STDOUT while building.
    ///
    /// This option is disabled by default.
    pub fn show_progress(mut self: Self, yes: bool) -> Self {
        self.show_progress = yes;
        self
    }
}

/// A builder for creating an index to be searched using `Granne`
pub struct GranneBuilder<Elements: ElementContainer> {
    elements: Elements,
    layers: Vec<FixedWidthSliceVector<'static, NeighborId>>,
    config: BuildConfig,
}

pub trait Builder: Index {
    // methods
    fn build(self: &mut Self);
    fn build_partial(self: &mut Self, num_elements: usize);
    fn num_elements(self: &Self) -> usize;
    fn write_index<B: std::io::Write + std::io::Seek>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<()>
    where
        Self: Sized;
}

impl<Elements: ElementContainer + Sync> Index for GranneBuilder<Elements> {
    // TODO: FIGURE OUT WHAT LEN MEANS FOR A BUILDER
    /// Returns the number of indexed elements.
    /// Note that it might be less than the number of elements in `elements`.
    /// # Examples
    /// ```
    /// # use granne::*;
    /// # let elements: angular::Vectors = test_helper::random_vectors(3, 1000);
    /// assert_eq!(1000, elements.len());
    /// let mut builder = GranneBuilder::new(BuildConfig::default(), elements);
    /// builder.build_partial(100);
    /// assert_eq!(100, builder.len());
    /// assert_eq!(1000, builder.num_elements());
    fn len(self: &Self) -> usize {
        self.get_index().len()
    }

    /// Returns the number of layers in this index.
    fn num_layers(self: &Self) -> usize {
        self.get_index().num_layers()
    }

    /// Returns the number of nodes in `layer`.
    fn layer_len(self: &Self, layer: usize) -> usize {
        self.get_index().layer_len(layer)
    }

    /// Returns the neighbors of the node at `index` in `layer`.
    fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize> {
        self.get_index().get_neighbors(index, layer)
    }
}

impl<Elements: ElementContainer + Sync> Builder for GranneBuilder<Elements> {
    /// Builds an index for approximate nearest neighbor search.
    fn build(self: &mut Self) {
        self.build_partial(self.elements.len())
    }

    /// Builds the search index for the first num_elements elements
    /// Can be used for long-running jobs where intermediate steps needs to be stored
    ///
    /// Note: already indexed elements are not reindexed
    fn build_partial(self: &mut Self, num_elements: usize) {
        if num_elements == 0 {
            return;
        }

        assert!(
            num_elements >= self.layers.last().map_or(0, |layer| layer.len()),
            "Cannot index fewer elements than already in index."
        );
        assert!(
            num_elements <= self.elements.len(),
            "Cannot index more elements than exist."
        );

        if !self.layers.is_empty() {
            self.index_elements_in_last_layer(num_elements);
        }

        while self.len() < num_elements {
            let new_layer = self.layers.last().map_or_else(
                || FixedWidthSliceVector::with_width(self.config.num_neighbors),
                |prev_layer| prev_layer.clone(),
            );

            self.layers.push(new_layer);

            self.index_elements_in_last_layer(num_elements);
        }
    }

    /// Returns the number of elements.
    fn num_elements(self: &Self) -> usize {
        self.elements.len()
    }

    /// Write the index to `buffer`.
    /// # Examples
    /// ```
    /// # use granne::*;
    /// # let builder = GranneBuilder::new(BuildConfig::default(), angular::Vectors::new());
    /// # let path = "/tmp/write_index.bin";
    /// let mut file = std::fs::File::create(path)?;
    /// builder.write_index(&mut file)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    fn write_index<B: std::io::Write + std::io::Seek>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<()> {
        let layers: Layers =
            Layers::FixWidth(self.layers.iter().map(|layer| layer.borrow()).collect());
        io::write_index(&layers, buffer)
    }
}

impl<Elements: ElementContainer + Sync> GranneBuilder<Elements> {
    /// Creates a new GranneBuilder with `elements`. The builder can be configured using `BuildConfig`.
    /// # Examples
    ///
    /// ```
    /// # use granne::*;
    /// let config = BuildConfig::default().num_neighbors(20).max_search(100).show_progress(true);
    /// let mut builder = GranneBuilder::new(config, angular::Vectors::new());
    /// ```
    pub fn new(config: BuildConfig, elements: Elements) -> Self {
        Self {
            elements,
            layers: Vec::new(),
            config,
        }
    }

    /*
       /// Creates a `GranneBuilder` by reading an already built index from `buffer` together with `elements`
       pub fn read(config: BuildConfig, buffer: &mut [u8], elements: Elements) -> Self {
           let mut builder = Self::new(config, elements);

           builder.layers = io::read_layers(buffer, builder.config.num_neighbors);

           builder
       }
    */

    // methods

    /// Returns a searchable index from this builder.
    /// # Examples
    /// ```
    /// # use granne::*;
    /// # let elements: angular::Vectors = test_helper::random_vectors(3, 1000);
    /// # let element = elements.get(123);
    /// # let max_search = 10; let num_neighbors = 20;
    /// let mut builder = GranneBuilder::new(BuildConfig::default(), elements);
    /// builder.build();
    /// let index = builder.get_index();
    /// index.search(&element, max_search, num_neighbors);
    /// ```
    pub fn get_index(self: &Self) -> Granne<&Elements> {
        Granne::from_parts(
            self.layers.iter().map(|l| l.borrow()).collect::<Vec<_>>(),
            &self.elements,
        )
    }

    /// Returns a reference to the elements in this builder.
    pub fn get_elements(self: &Self) -> &Elements {
        &self.elements
    }
}

impl<Elements: ElementContainer + crate::io::Writeable> GranneBuilder<Elements> {
    /// Write the elements of this builder to `buffer`.
    /// # Examples
    /// ```
    /// # use granne::*;
    /// # let builder = GranneBuilder::new(BuildConfig::default(), angular::Vectors::new());
    /// # let path = "/tmp/write_elements.bin";
    /// let mut file = std::fs::File::create(path)?;
    /// builder.write_elements(&mut file)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn write_elements<B: std::io::Write>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<usize> {
        self.elements.write(buffer)
    }
}

impl<Elements: ExtendableElementContainer> GranneBuilder<Elements> {
    /// Push a new element into this builder. In order to insert it into the index
    /// a call to `build` or `build_part` is required.
    /// # Examples
    /// ```
    /// # use granne::*;
    /// # let element0 = test_helper::random_vector(3);
    /// # let element1 = test_helper::random_vector(3);
    /// let mut builder = GranneBuilder::new(BuildConfig::default(), angular::Vectors::new());
    /// builder.push(element0);
    /// builder.push(element1);
    /// assert_eq!(0, builder.len());
    /// builder.build();
    /// assert_eq!(2, builder.len());
    /// ```
    pub fn push(self: &mut Self, element: Elements::InternalElement) {
        self.elements.push(element);
    }
}

// implementation

trait Graph {
    fn get_neighbors(self: &Self, idx: usize) -> Vec<usize>;
    fn len(self: &Self) -> usize;
}

impl<'a> Graph for FixedWidthSliceVector<'a, NeighborId> {
    fn get_neighbors(self: &Self, idx: usize) -> Vec<usize> {
        self.get(idx)
            .iter()
            .take_while(|&&x| x != UNUSED)
            .map(|&x| usize::try_from(x).unwrap())
            .collect()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a> Graph for MultiSetVector<'a> {
    fn get_neighbors(self: &Self, idx: usize) -> Vec<usize> {
        self.get(idx)
            .iter()
            .map(|&x| usize::try_from(x).unwrap())
            .collect()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a> Graph for [parking_lot::RwLock<&'a mut [NeighborId]>] {
    fn get_neighbors(self: &Self, idx: usize) -> Vec<usize> {
        self[idx]
            .read()
            .iter()
            .take_while(|&&x| x != UNUSED)
            .map(|&x| usize::try_from(x).unwrap())
            .collect()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

enum Layers<'a> {
    FixWidth(Vec<FixedWidthSliceVector<'a, NeighborId>>),
    Compressed(Vec<MultiSetVector<'a>>),
}

impl<'a> Layers<'a> {
    fn len(self: &Self) -> usize {
        match self {
            Self::FixWidth(layers) => layers.len(),
            Self::Compressed(layers) => layers.len(),
        }
    }

    fn as_graph(self: &Self, layer: usize) -> &dyn Graph {
        match self {
            Self::FixWidth(layers) => &layers[layer],
            Self::Compressed(layers) => &layers[layer],
        }
    }
}

impl<'a> From<Vec<FixedWidthSliceVector<'a, NeighborId>>> for Layers<'a> {
    fn from(fix_width: Vec<FixedWidthSliceVector<'a, NeighborId>>) -> Self {
        Self::FixWidth(fix_width)
    }
}

/// Computes the number of elements that should be in layer `layer_idx`.
fn compute_num_elements_in_layer(
    total_num_elements: usize,
    layer_multiplier: f32,
    layer_idx: usize,
) -> usize {
    cmp::min(
        (total_num_elements as f32
            / (layer_multiplier.powf(
                (total_num_elements as f32).log(layer_multiplier).floor() - layer_idx as f32,
            )))
        .ceil() as usize,
        total_num_elements,
    )
}

impl<Elements: ElementContainer + Sync> GranneBuilder<Elements> {
    fn index_elements_in_last_layer(self: &mut Self, max_num_elements: usize) {
        let total_num_elements = self
            .config
            .expected_num_elements
            .unwrap_or(self.elements.len());
        let ideal_num_elements_in_layer = compute_num_elements_in_layer(
            total_num_elements,
            self.config.layer_multiplier,
            self.layers.len() - 1,
        );
        let num_elements_in_layer = cmp::min(max_num_elements, ideal_num_elements_in_layer);
        let additional = ideal_num_elements_in_layer - self.layers.last().unwrap().len();

        if additional == 0 {
            // nothing to index in this layer
            return;
        }

        let mut config = self.config.clone();

        // if not last layer
        if ideal_num_elements_in_layer < total_num_elements {
            // use half num_neighbors on upper layers
            config.num_neighbors = cmp::max(1, config.num_neighbors / 2);
        }

        let mut layer = self.layers.pop().unwrap();

        layer.reserve_exact(additional);

        let prev_layers = self.get_index();

        if self.config.show_progress {
            println!(
                "Building layer {} with {} elements...",
                self.layers.len(),
                num_elements_in_layer
            );
        }

        Self::index_elements(
            &config,
            &self.elements,
            num_elements_in_layer,
            &prev_layers,
            &mut layer,
            false,
        );

        if self.config.reinsert_elements {
            if self.config.show_progress {
                println!("Reinserting elements...");
            }

            // use half max_search when reindexing
            config.max_search = std::cmp::max(1, config.max_search / 2);

            // reinsert elements to improve index quality
            Self::index_elements(
                &config,
                &self.elements,
                num_elements_in_layer,
                &prev_layers,
                &mut layer,
                true,
            );
        }

        self.layers.push(layer);
    }

    /// Indexes elements in `layer`.
    fn index_elements(
        config: &BuildConfig,
        elements: &Elements,
        num_elements: usize,
        prev_layers: &Granne<&Elements>,
        layer: &mut FixedWidthSliceVector<'static, NeighborId>,
        reinsert_elements: bool,
    ) {
        assert!(layer.len() <= num_elements);

        let mut already_indexed = layer.len();
        if reinsert_elements {
            already_indexed = 0;
        } else {
            layer.resize(num_elements, UNUSED);
        }

        // set up progress bar
        let step_size = cmp::max(100, num_elements / 400);
        let (progress_bar, start_time) = {
            if config.show_progress {
                let mut progress_bar = pbr::ProgressBar::new(elements.len() as u64);
                let info_text = format!("Layer {}: ", prev_layers.layers.len());
                progress_bar.message(&info_text);
                progress_bar.set((step_size * (already_indexed / step_size)) as u64);

                // if too many elements were already indexed, the shown speed
                // is misrepresenting and not of much help
                if already_indexed > num_elements / 3 {
                    progress_bar.show_speed = false;
                    progress_bar.show_time_left = false;
                }

                (
                    Some(parking_lot::Mutex::new(progress_bar)),
                    Some(time::PreciseTime::now()),
                )
            } else {
                (None, None)
            }
        };

        {
            // create RwLocks for underlying nodes
            let layer: Vec<parking_lot::RwLock<&mut [NeighborId]>> =
                layer.iter_mut().map(parking_lot::RwLock::new).collect();

            let insert_element = |(idx, _)| {
                Self::index_element(config, elements, prev_layers, &layer, idx);

                // This only shows approximate progress because of par_iter
                if idx % step_size == 0 {
                    if let Some(ref progress_bar) = progress_bar {
                        progress_bar.lock().add(step_size as u64);
                    }
                }
            };

            #[cfg(feature = "singlethreaded")]
            let layer_iter = layer.iter();
            #[cfg(not(feature = "singlethreaded"))]
            let layer_iter = layer.par_iter();

            if reinsert_elements {
                // reinserting elements is done in reverse order
                layer_iter.enumerate().rev().for_each(insert_element);
            } else {
                // index elements, skipping already indexed
                layer_iter
                    .enumerate()
                    .skip(already_indexed)
                    .for_each(insert_element);
            };
        }

        if let Some(progress_bar) = progress_bar {
            progress_bar.lock().set(layer.len() as u64);
        }

        #[cfg(feature = "singlethreaded")]
        let layer_iter_mut = layer.iter_mut();
        #[cfg(not(feature = "singlethreaded"))]
        let layer_iter_mut = layer.par_iter_mut();

        // limit number of neighbors (i.e. apply heuristic for neighbor selection)
        layer_iter_mut.enumerate().for_each(|(i, node)| {
            Self::add_and_limit_neighbors(elements, node, i, &[], config.num_neighbors);
        });

        if let Some(start_time) = start_time {
            let end_time = time::PreciseTime::now();
            println!("Time: {} s", start_time.to(end_time).num_seconds());
        }
    }

    /// Indexes the element with index `idx`.
    fn index_element(
        config: &BuildConfig,
        elements: &Elements,
        prev_layers: &Granne<&Elements>,
        layer: &[parking_lot::RwLock<&mut [NeighborId]>],
        idx: usize,
    ) {
        // do not index elements that are zero
        if elements.dist(idx, idx) > NotNan::new(0.0001).unwrap() {
            // * Element::eps() {
            return;
        }

        let element = elements.get(idx);

        let entrypoint = prev_layers
            .search(&element, 1, 1)
            .first()
            .map_or(0, |r| r.0);
        let candidates =
            search_for_neighbors(layer, entrypoint, elements, &element, config.max_search);

        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|&(id, _)| id != idx)
            .collect();

        let neighbors = Self::select_neighbors(elements, candidates, config.num_neighbors);

        // if the current element is a duplicate of too many of its potential neighbors, do not connect it to the graph,
        // this effectively creates a dead node
        if let Some((_, d)) = neighbors.get(config.num_neighbors / 2) {
            if *d < NotNan::new(0.000001).unwrap() {
                return;
            }
        }

        // if current node is empty, initialize it with the neighbors
        if layer[idx].read()[0] == UNUSED {
            Self::initialize_node(&layer[idx], &neighbors[..]);
        } else {
            for &(neighbor, d) in &neighbors {
                Self::connect_nodes(elements, &layer[idx], idx, neighbor, d);
            }
        }

        for (neighbor, d) in neighbors {
            Self::connect_nodes(elements, &layer[neighbor], neighbor, idx, d);
        }
    }

    /// Given a vec of `candidates`, selects the neighbors for an element.
    ///
    fn select_neighbors(
        elements: &Elements,
        candidates: Vec<(usize, NotNan<f32>)>,
        max_neighbors: usize,
    ) -> Vec<(usize, NotNan<f32>)> {
        if candidates.len() <= max_neighbors {
            return candidates;
        }

        // candidates need to be sorted on distance from idx
        debug_assert!(candidates
            .iter()
            .zip(candidates.iter().skip(1))
            .all(|((_, d0), (_, d1))| d0 <= d1));

        let mut neighbors: Vec<(usize, NotNan<f32>)> = Vec::new();

        for (j, d) in candidates {
            if neighbors.len() >= max_neighbors {
                break;
            }

            // add j to neighbors if j is closer to idx,
            // than to all previously added neighbors
            let element = elements.get(j);
            if neighbors
                .iter()
                .all(|&(n, _)| d <= elements.dist_to_element(n, &element))
            {
                neighbors.push((j, d));
            }
        }

        neighbors
    }

    /// Sets neighbors for `node`.
    fn initialize_node(
        node: &parking_lot::RwLock<&mut [NeighborId]>,
        neighbors: &[(usize, NotNan<f32>)],
    ) {
        // Write Lock!
        let mut node = node.write();

        debug_assert_eq!(UNUSED, node[0]);
        for (i, &(idx, _)) in neighbors.iter().enumerate().take(node.len()) {
            node[i] = NeighborId::try_from(idx).unwrap();
        }
    }

    /// Tries to add `j` as a neighbor to `i`. If the neighbor list is full, uses `select_neighbors`
    /// to limit the number of neighbors.
    fn connect_nodes(
        elements: &Elements,
        node: &parking_lot::RwLock<&mut [NeighborId]>,
        i: usize,
        j: usize,
        d: NotNan<f32>,
    ) {
        if i == j {
            return;
        }

        // Write Lock!
        let mut node = node.write();

        // Do not insert duplicates
        let j_id = NeighborId::try_from(j).unwrap();
        if let Some(free_pos) = node.iter().position(|x| *x == UNUSED || *x == j_id) {
            node[free_pos] = j_id;
        } else {
            let num_neighbors = node.len();
            Self::add_and_limit_neighbors(elements, &mut node, i, &[(j, d)], num_neighbors);
        }
    }

    fn add_and_limit_neighbors(
        elements: &Elements,
        node: &mut [NeighborId],
        node_id: usize,
        extra: &[(usize, NotNan<f32>)],
        num_neighbors: usize,
    ) {
        assert!(num_neighbors <= node.len());

        let neighbors: Vec<usize> = node
            .iter()
            .take_while(|&&x| x != UNUSED)
            .map(|&x| usize::try_from(x).unwrap())
            .collect();

        let dists = elements.dists(node_id, &neighbors);
        let mut candidates: Vec<_> = neighbors.iter().copied().zip(dists.into_iter()).collect();

        for &(j, d) in extra {
            candidates.push((j, d));
        }

        candidates.sort_unstable_by_key(|&(_, d)| d);

        let neighbors = Self::select_neighbors(elements, candidates, num_neighbors);

        // set new neighbors and mark last positions as unused
        for (k, n) in neighbors
            .into_iter()
            .map(|(n, _)| NeighborId::try_from(n).unwrap())
            .chain(std::iter::repeat(UNUSED))
            .enumerate()
            .take(node.len())
        {
            node[k] = n;
        }
    }
}

impl<'a, Elements: ElementContainer> Granne<'a, Elements> {
    fn from_parts<L: Into<Layers<'a>>>(layers: L, elements: Elements) -> Self {
        Self {
            layers: layers.into(),
            elements,
        }
    }

    fn search_internal(
        self: &Self,
        layers: &[impl Graph],
        element: &Elements::Element,
        max_search: usize,
        num_neighbors: usize,
    ) -> Vec<(usize, f32)> {
        if let Some((bottom_layer, top_layers)) = layers.split_last() {
            let entrypoint = find_entrypoint(top_layers, &self.elements, element);

            search_for_neighbors(
                bottom_layer,
                entrypoint,
                &self.elements,
                element,
                max_search,
            )
            .into_iter()
            .take(num_neighbors)
            .map(|(i, d)| (i, d.into_inner()))
            .collect()
        } else {
            Vec::new()
        }
    }
}

fn find_entrypoint<Layer: Graph, Elements: ElementContainer>(
    layers: &[Layer],
    elements: &Elements,
    element: &Elements::Element,
) -> usize {
    let mut entrypoint = 0;
    for layer in layers {
        let res = search_for_neighbors(layer, entrypoint, elements, element, 1);

        entrypoint = res[0].0;
    }

    entrypoint
}

fn search_for_neighbors<Layer: Graph + ?Sized, Elements: ElementContainer>(
    layer: &Layer,
    entrypoint: usize,
    elements: &Elements,
    goal: &Elements::Element,
    max_search: usize,
) -> Vec<(usize, NotNan<f32>)> {
    let mut res: max_size_heap::MaxSizeHeap<(NotNan<f32>, usize)> =
        max_size_heap::MaxSizeHeap::new(max_search); // TODO: should this really be max_search or num_neighbors?
    let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();

    let num_neighbors = 20; //layer.at(0).len();
    let mut visited =
        HashSet::with_capacity_and_hasher(max_search * num_neighbors, FxBuildHasher::default());

    let distance = elements.dist_to_element(entrypoint, goal);

    pq.push(RevOrd((distance, entrypoint)));

    visited.insert(entrypoint);

    while let Some(RevOrd((d, idx))) = pq.pop() {
        if res.is_full() && d > res.peek().unwrap().0 {
            break;
        }

        res.push((d, idx));

        for neighbor_idx in layer.get_neighbors(idx) {
            if visited.insert(neighbor_idx) {
                let distance = elements.dist_to_element(neighbor_idx, goal);

                if !res.is_full() || distance < res.peek().unwrap().0 {
                    pq.push(RevOrd((distance, neighbor_idx)));
                }
            }
        }
    }

    res.into_sorted_vec()
        .into_iter()
        .map(|(d, idx)| (idx, d))
        .collect()
}
