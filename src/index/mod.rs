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

use crate::{
    max_size_heap,
    slice_vector::{FixedWidthSliceVector, MultiSetVector, VariableWidthSliceVector},
    {ElementContainer, ExtendableElementContainer},
};

type NeighborId = u32;
const UNUSED: NeighborId = NeighborId::max_value();

/// An index for fast approximate nearest neighbor search.
/// The index is built by using `GranneBuilder` and can be stored to disk.
pub struct Granne<'a, Elements: ElementContainer> {
    layers: Layers<'a>,
    elements: &'a Elements,
}

impl<'a, Elements: ElementContainer> Granne<'a, Elements> {
    /// Loads this index lazily
    pub fn load(index: &'a [u8], elements: &'a Elements) -> Self {
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
            Layers::VarWidth(layers) => {
                self.search_internal(&layers, element, max_search, num_neighbors)
            }
            Layers::Compressed(layers) => {
                self.search_internal(&layers, element, max_search, num_neighbors)
            }
        }
    }

    /// Returns the number of elements in this index.
    /// Note that it might be less than the number of elements in `elements`.
    pub fn len(self: &Self) -> usize {
        match &self.layers {
            Layers::FixWidth(layers) => layers.last().map(|l| l.len()).unwrap_or(0),
            Layers::VarWidth(layers) => layers.last().map(|l| l.len()).unwrap_or(0),
            Layers::Compressed(layers) => layers.last().map(|l| l.len()).unwrap_or(0),
        }
    }

    /// Returns the number of layers in this index.
    pub fn num_layers(self: &Self) -> usize {
        self.layers.len()
    }

    /// Returns the number of nodes in `layer`.
    pub fn layer_len(self: &Self, layer: usize) -> usize {
        match &self.layers {
            Layers::FixWidth(layers) => layers[layer].len(),
            Layers::VarWidth(layers) => layers[layer].len(),
            Layers::Compressed(layers) => layers[layer].len(),
        }
    }

    /// Returns the element at `index`.
    pub fn get_element(self: &Self, index: usize) -> Elements::Element {
        self.elements.get(index)
    }

    /// Returns the neighbors of the node at `index` in `layer`.
    pub fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize> {
        match &self.layers {
            Layers::FixWidth(layers) => layers[layer].get_neighbors(index),
            Layers::VarWidth(layers) => layers[layer].get_neighbors(index),
            Layers::Compressed(layers) => layers[layer].get_neighbors(index),
        }
    }

    /// Returns a Granne index with the nodes reordered according to the permutation `order`.
    ///
    /// `order[i] == j`, means that the element with idx `j`, will be moved to idx `i`.
    /// `order` must respect the layers ----.
    pub fn reordered_index<'b>(
        self: &Self,
        order: &[usize],
        reordered_elements: &'b Elements,
        show_progress: bool,
    ) -> Granne<'b, Elements> {
        Granne {
            layers: reorder::reorder_layers(&self.layers, order, show_progress),
            elements: reordered_elements,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    /// Number of layers in the final graph. Each new layer will have exponentially more nodes than the one below. E.g.
    /// layer 0: 1 node, layer 1: 10 nodes, layer 2: 100 nodes, ...
    ///
    /// Choosing the number layers so that the .. (use layer multiplier instead)
    pub num_layers: usize,
    //pub layer_multiplier: f32,
    /// The maximum number of neighbors per node and layer.
    pub num_neighbors: usize,

    /// The `max_search` parameter used during build time (see `granne::search`).
    pub max_search: usize,

    /// Whether to reinsert all the elements in each layers. Takes more time, but improves recall.
    pub reinsert_elements: bool,

    /// Whether to output progress information to STDOUT while building.
    pub show_progress: bool,
}

/// A builder for creating an index to be searched using `Granne`
pub struct GranneBuilder<Elements: ElementContainer> {
    elements: Elements,
    layers: Vec<FixedWidthSliceVector<'static, NeighborId>>,
    config: Config,
}

impl<Elements: ElementContainer + Sync> GranneBuilder<Elements> {
    pub fn new(config: Config, elements: Elements) -> Self {
        Self {
            elements,
            layers: Vec::new(),
            config,
        }
    }
    /*
        /// Creates a `GranneBuilder` by reading an already built index from `buffer` together with `elements`
        pub fn read(config: Config, buffer: &mut [u8], elements: Elements) -> Self {
            let mut builder = Self::new(config, elements);

            builder.layers = io::read_layers(buffer, builder.config.num_neighbors);

            builder
        }
    */
    /// Returns the number of already indexed elements.
    pub fn indexed_elements(self: &Self) -> usize {
        self.layers.last().map_or(0, |l| l.len())
    }

    /// Returns a searchable index from this builder.
    pub fn get_index(self: &Self) -> Granne<Elements> {
        Granne::from_parts(
            self.layers.iter().map(|l| l.borrow()).collect::<Vec<_>>(),
            &self.elements,
        )
    }

    /// Returns a reference to the elements in this builder.
    pub fn get_elements(self: &Self) -> &Elements {
        &self.elements
    }

    /// Builds an index for approximate nearest neighbor search.
    pub fn build_index(&mut self) {
        self.build_index_part(self.elements.len())
    }

    /// Builds the search index for the first num_elements elements
    /// Can be used for long-running jobs where intermediate steps needs to be stored
    ///
    /// Note: already indexed elements are not reindexed
    pub fn build_index_part(&mut self, num_elements: usize) {
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

        // fresh index build => initialize first layer
        if self.layers.is_empty() {
            let mut layer = FixedWidthSliceVector::with_width(self.config.num_neighbors);
            layer.resize(1, UNUSED);
            self.layers.push(layer);
        } else {
            // make sure the current last layer is full (no-op if already full)
            self.index_elements_in_last_layer(num_elements);
        }

        // push new layers
        for _layer in self.layers.len()..self.config.num_layers {
            if num_elements == self.indexed_elements() {
                // already done
                break;
            }

            // create a new layer by copying the current last one
            let new_layer = self.layers.last().expect("There are no layers!").clone();
            self.layers.push(new_layer);

            self.index_elements_in_last_layer(num_elements);
        }
    }

    /// Write the index to `buffer`.
    pub fn write_index<B: std::io::Write + std::io::Seek>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<()> {
        io::write_index(&self.layers, buffer)
    }
}

impl<Elements: ElementContainer + crate::io::Writeable> GranneBuilder<Elements> {
    /// Write the elements of this builder to `buffer`.
    pub fn write_elements<B: std::io::Write>(
        self: &Self,
        buffer: &mut B,
    ) -> std::io::Result<usize> {
        self.elements.write(buffer)
    }
}

impl<Elements: ExtendableElementContainer> GranneBuilder<Elements> {
    /// Push a new element into this builder. In order to insert it into the index
    /// a call to `build_index` or `build_index_part` is required.
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

impl<'a> Graph for VariableWidthSliceVector<'a, NeighborId, usize> {
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

pub enum Layers<'a> {
    FixWidth(Vec<FixedWidthSliceVector<'a, NeighborId>>),
    VarWidth(Vec<VariableWidthSliceVector<'a, NeighborId, usize>>),
    Compressed(Vec<MultiSetVector<'a>>),
}

impl<'a> Layers<'a> {
    fn len(self: &Self) -> usize {
        match self {
            Self::FixWidth(layers) => layers.len(),
            Self::VarWidth(layers) => layers.len(),
            Self::Compressed(layers) => layers.len(),
        }
    }
}

impl<'a> From<Vec<FixedWidthSliceVector<'a, NeighborId>>> for Layers<'a> {
    fn from(fix_width: Vec<FixedWidthSliceVector<'a, NeighborId>>) -> Self {
        Self::FixWidth(fix_width)
    }
}

/// Computes a layer multiplier `m`, s.t. the number of elements in layer `i` is
/// equal to `m^i` and `m^num_layers ~= num_elements`
fn compute_layer_multiplier(num_elements: usize, num_layers: usize) -> f32 {
    (num_elements as f32).powf(1.0 / (num_layers - 1) as f32)
}

impl<Elements: ElementContainer + Sync> GranneBuilder<Elements> {
    fn index_elements_in_last_layer(self: &mut Self, max_num_elements: usize) {
        /*        let layer_multiplier = self
           .layer_multiplier
           .unwrap_or(compute_layer_multiplier(self.elements.len(), self.config.num_layers));
        */

        let layer_multiplier =
            compute_layer_multiplier(self.elements.len(), self.config.num_layers);

        let layer = self.layers.len() - 1;
        let ideal_num_elements_in_layer = cmp::min(
            layer_multiplier.powf(layer as f32).ceil() as usize,
            self.elements.len(),
        );
        let mut num_elements_in_layer = cmp::min(max_num_elements, ideal_num_elements_in_layer);

        let mut config = self.config.clone();

        if layer == self.config.num_layers - 1 {
            // if last layer index all elements
            num_elements_in_layer = max_num_elements;
        } else {
            // use half num_neighbors on upper layers
            config.num_neighbors = cmp::max(1, config.num_neighbors / 2);
        }

        let additional = ideal_num_elements_in_layer - self.layers.last().unwrap().len();

        if additional == 0 {
            // nothing to index in this layer
            return;
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

    fn index_elements(
        config: &Config,
        elements: &Elements,
        num_elements: usize,
        prev_layers: &Granne<Elements>,
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

            if reinsert_elements {
                // reinserting elements is done in reverse order
                layer.par_iter().enumerate().rev().for_each(insert_element);
            } else {
                // index elements, skipping already indexed
                layer
                    .par_iter()
                    .enumerate()
                    .skip(already_indexed)
                    .for_each(insert_element);
            };
        }

        if let Some(progress_bar) = progress_bar {
            progress_bar.lock().set(layer.len() as u64);
        }

        // limit number of neighbors (i.e. apply heuristic for neighbor selection)
        layer.par_iter_mut().enumerate().for_each(|(i, node)| {
            Self::add_and_limit_neighbors(elements, node, i, &[], config.num_neighbors);
        });

        if let Some(start_time) = start_time {
            let end_time = time::PreciseTime::now();
            println!("Time: {} s", start_time.to(end_time).num_seconds());
        }
    }

    fn index_element<'a>(
        config: &Config,
        elements: &Elements,
        prev_layers: &Granne<Elements>,
        layer: &'a [parking_lot::RwLock<&'a mut [NeighborId]>],
        idx: usize,
    ) {
        // do not index elements that are zero
        if elements.dist(idx, idx) > NotNan::new(0.0001).unwrap() {
            // * Element::eps() {
            return;
        }

        let element = elements.get(idx);

        if let Some((entrypoint, _)) = prev_layers.search(&element, 1, 1).first() {
            let candidates =
                search_for_neighbors(layer, *entrypoint, elements, &element, config.max_search);

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
    }

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
            if neighbors.iter().all(|&(n, _)| d <= elements.dist(n, j)) {
                neighbors.push((j, d));
            }
        }

        neighbors
    }

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
    fn from_parts<L: Into<Layers<'a>>>(layers: L, elements: &'a Elements) -> Self {
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
            let entrypoint = find_entrypoint(top_layers, self.elements, element);

            search_for_neighbors(bottom_layer, entrypoint, self.elements, element, max_search)
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
        max_size_heap::MaxSizeHeap::new(max_search);
    let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();

    let num_neighbors = 20; //layer.at(0).len();
    let mut visited = HashSet::with_capacity(max_search * num_neighbors);

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
