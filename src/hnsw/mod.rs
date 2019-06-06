use hashbrown;
use ordered_float::NotNaN;
use parking_lot::{Mutex, RwLock};
use pbr::ProgressBar;
use rayon::prelude::*;
use revord::RevOrd;
use std::borrow::Cow;
use std::cmp;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Read, Result, Write};
use time::PreciseTime;

use slice_vector::{FixedWidthSliceVector, VariableWidthSliceVector};

use crate::file_io;
use crate::types::ComparableTo;

mod io;
mod neighborid;
pub mod reorder;
pub mod rw_builder;
mod sharded_hnsw;
#[cfg(test)]
mod tests;

pub use self::io::compress_index;
pub use self::sharded_hnsw::ShardedHnsw;

use self::neighborid::{NeighborId, UNUSED};

const METADATA_LEN: usize = 1024;
const LIBRARY_STR: &str = "granne";
const SERIALIZATION_VERSION: usize = 1;

type HnswNode = [NeighborId];

fn iter_neighbors<'b>(node: &'b HnswNode) -> impl 'b + Iterator<Item = usize> {
    node.iter().take_while(|&&n| n != UNUSED).map(|&n| n.into())
}

#[derive(Clone)]
pub struct Config {
    pub num_layers: usize,
    pub num_neighbors: usize,
    pub max_search: usize,
    pub reinsert_elements: bool,
    pub show_progress: bool,
}

impl Config {
    pub fn default() -> Self {
        Self {
            num_layers: 7,
            num_neighbors: 20,
            max_search: 200,
            reinsert_elements: true,
            show_progress: true,
        }
    }
}

pub struct HnswBuilder<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + Sync + Send + ToOwned + ?Sized,
    Element: 'a + ComparableTo<Element> + Sync + Send,
{
    layers: Vec<FixedWidthSliceVector<'static, NeighborId>>,
    elements: Cow<'a, Elements>,
    config: Config,
}

/// The At trait is similar to std::ops::Index. The latter however always returns a reference which makes it impossible
/// to implement it for containers where the elements are stored compressed and the Output is only temporarily created.
pub trait At {
    type Output;
    fn at(self: &Self, index: usize) -> Self::Output;
    fn len(self: &Self) -> usize;
}

// Implement At for all slices of cloneable objects. Using this instead of normal indexing (std::ops::Index) is associated
// with a small performance penalty (since each element needs to be cloned on access).
impl<T: Clone> At for [T] {
    type Output = T;

    fn at(self: &Self, index: usize) -> Self::Output {
        self[index].clone()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

pub trait Writeable {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()>;
}

impl<T> Writeable for [T] {
    fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        file_io::write(self, buffer)
    }
}

pub trait Appendable {
    type Element;
    fn new() -> Self;
    fn append(self: &mut Self, element: Self::Element);
}

impl<T> Appendable for Vec<T> {
    type Element = T;

    fn new() -> Self {
        Vec::new()
    }

    fn append(self: &mut Self, element: T) {
        self.push(element);
    }
}

pub trait Search {
    type Element;
    fn search(self: &Self, element: &Self::Element, num_neighbors: usize, max_search: usize) -> Vec<(usize, f32)>;
}

impl<'a, Elements, Element> HnswBuilder<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + Sync + Send + ToOwned + ?Sized,
    Element: 'a + ComparableTo<Element> + Sync + Send,
{
    pub fn with_borrowed_elements(config: Config, elements: &'a Elements) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Borrowed(elements),
            config: config,
        }
    }

    pub fn with_owned_elements(config: Config, elements: Elements::Owned) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Owned(elements),
            config: config,
        }
    }

    pub fn read_index_with_owned_elements<I: Read>(
        config: Config,
        index_reader: &mut I,
        elements: Elements::Owned,
    ) -> Result<Self> {
        let elements: Cow<Elements> = Cow::Owned(elements);

        let layers = io::read_layers(index_reader, Some((config.num_layers, elements.len())))?;

        if let Some(ref last_layer) = layers.last() {
            assert!(last_layer.len() <= elements.len());
        }

        Ok(Self {
            layers: layers,
            elements: elements,
            config: config,
        })
    }

    pub fn read_index_with_borrowed_elements<I: Read>(
        config: Config,
        index_reader: &mut I,
        elements: &'a Elements,
    ) -> Result<Self> {
        let layers = io::read_layers(index_reader, Some((config.num_layers, elements.len())))?;

        if let Some(ref last_layer) = layers.last() {
            assert!(last_layer.len() <= elements.len());
        }

        Ok(Self {
            layers: layers,
            elements: Cow::Borrowed(elements),
            config: config,
        })
    }

    pub fn len(self: &Self) -> usize {
        self.elements.len()
    }

    pub fn indexed_elements(self: &Self) -> usize {
        if let Some(layer) = self.layers.last() {
            layer.len()
        } else {
            0
        }
    }

    pub fn save_index_to_disk(self: &Self, path: &str, compress: bool) -> Result<()> {
        self.get_index().save_index_to_disk(path, compress)
    }

    pub fn write(self: &Self, file: &mut File) -> Result<()> {
        io::save_index_to_disk(&self.get_index().layers, file, false)
    }

    pub fn get_index<'b>(self: &'b Self) -> Hnsw<'b, Elements, Element> {
        Hnsw {
            layers: Layers::FixWidth(self.layers.iter().map(|layer| layer.borrow()).collect()),
            elements: self.elements.as_ref(),
        }
    }

    /// Builds the search index for all elements
    pub fn build_index(&mut self) {
        let num_elements = self.elements.len();
        self.build_index_part(num_elements);
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
            let mut layer = FixedWidthSliceVector::new(self.config.num_neighbors);
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
            let new_layer = self.layers.last().unwrap().clone();
            self.layers.push(new_layer);

            self.index_elements_in_last_layer(num_elements);
        }
    }

    fn index_elements_in_last_layer(&mut self, max_num_elements: usize) {
        let layer_multiplier = compute_layer_multiplier(self.elements.len(), self.config.num_layers);

        let layer = self.layers.len() - 1;
        let ideal_num_elements_in_layer =
            cmp::min(layer_multiplier.powf(layer as f32).ceil() as usize, self.elements.len());
        let mut num_elements_in_layer = cmp::min(max_num_elements, ideal_num_elements_in_layer);

        let mut config = self.config.clone();

        if layer == self.config.num_layers - 1 {
            // if last layer index all elements
            num_elements_in_layer = max_num_elements;
        } else {
            // use half num_neighbors on upper layers
            config.num_neighbors = std::cmp::max(1, config.num_neighbors / 2);
        }

        let mut layer = self.layers.pop().unwrap();

        let additional = ideal_num_elements_in_layer - layer.len();
        layer.reserve_exact(additional);

        let prev_layers = self.get_index();

        if self.config.show_progress {
            println!(
                "Building layer {} with {} elements...",
                prev_layers.num_layers(),
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
        prev_layers: &Hnsw<Elements, Element>,
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
                let mut progress_bar = ProgressBar::new(elements.len() as u64);
                let info_text = format!("Layer {}: ", prev_layers.num_layers());
                progress_bar.message(&info_text);
                progress_bar.set((step_size * (already_indexed / step_size)) as u64);

                // if too many elements were already indexed, the shown speed
                // is misrepresenting and not of much help
                if already_indexed > num_elements / 3 {
                    progress_bar.show_speed = false;
                    progress_bar.show_time_left = false;
                }

                (Some(Mutex::new(progress_bar)), Some(PreciseTime::now()))
            } else {
                (None, None)
            }
        };

        {
            // create RwLocks for underlying nodes
            let layer: Vec<RwLock<&mut HnswNode>> = layer.iter_mut().map(|node| RwLock::new(node)).collect();

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
            let end_time = PreciseTime::now();
            println!("Time: {} s", start_time.to(end_time).num_seconds());
        }
    }

    fn index_element<Index: Search<Element = Element>>(
        config: &Config,
        elements: &Elements,
        prev_layers: &Index,
        layer: &[RwLock<&mut HnswNode>],
        idx: usize,
    ) {
        let element: Element = elements.at(idx);

        // do not index elements that are zero (multiply by 100 as safety margin)
        if element.dist(&element) > NotNaN::new(100.0).unwrap() * Element::eps() {
            return;
        }

        if let Some((entrypoint, _)) = prev_layers.search(&element, 1, 1).first() {
            let candidates =
                Self::search_for_neighbors_index(elements, layer, *entrypoint, &element, config.max_search);
            let candidates: Vec<_> = candidates.into_iter().filter(|&(id, _)| id != idx).collect();

            let neighbors = Self::select_neighbors(elements, candidates, config.num_neighbors);

            // if the current element is a duplicate of too many of its potential neighbors, do not connect it to the graph,
            // this effectively creates a dead node
            if let Some((_, d)) = neighbors.get(config.num_neighbors / 2) {
                if *d < Element::eps() {
                    return;
                }
            }

            // if current node is empty, initialize it with the neighbors
            let unused: usize = UNUSED.into();
            if iter_neighbors(&layer[idx].read()).next().unwrap_or(unused) == unused {
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

    // Similar to Hnsw::search_for_neighbors but with RwLocks for
    // parallel indexing
    fn search_for_neighbors_index(
        elements: &Elements,
        layer: &[RwLock<&mut HnswNode>],
        entrypoint: usize,
        goal: &Element,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {
        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();

        let num_neighbors = layer[entrypoint].read().len();
        let mut visited = hashbrown::HashSet::with_capacity(max_search * num_neighbors);

        pq.push(RevOrd((elements.at(entrypoint).dist(goal), entrypoint)));

        visited.insert(entrypoint);

        while let Some(RevOrd((d, idx))) = pq.pop() {
            if res.is_full() && d > res.peek().unwrap().0 {
                break;
            }

            res.push((d, idx));

            let node = layer[idx].read();

            for neighbor_idx in iter_neighbors(&node) {
                if visited.insert(neighbor_idx) {
                    let distance = elements.at(neighbor_idx).dist(goal);

                    if !res.is_full() || distance < res.peek().unwrap().0 {
                        pq.push(RevOrd((distance, neighbor_idx)));
                    }
                }
            }
        }

        res.heap
            .into_sorted_vec()
            .into_iter()
            .map(|(d, idx)| (idx, d))
            .collect()
    }

    fn select_neighbors(
        elements: &Elements,
        candidates: Vec<(usize, NotNaN<f32>)>,
        max_neighbors: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {
        if candidates.len() <= max_neighbors {
            return candidates;
        }

        let mut neighbors: Vec<(usize, NotNaN<f32>, Element)> = Vec::new();
        let mut pruned = Vec::new();

        // candidates are sorted on distance from idx
        for (j, d) in candidates.into_iter() {
            if neighbors.len() >= max_neighbors {
                break;
            }

            let element: Element = elements.at(j);

            // add j to neighbors if j is closer to idx,
            // than to all previously added neighbors
            if neighbors
                .iter()
                .all(|&(_, _, ref neighbor)| d <= neighbor.dist(&element) + Element::eps())
            {
                neighbors.push((j, d, element));
            } else {
                pruned.push((j, d));
            }
        }

        let neighbors: Vec<_> = neighbors.into_iter().map(|(j, d, _)| (j, d)).collect();

        neighbors
    }

    fn initialize_node(node: &RwLock<&mut HnswNode>, neighbors: &[(usize, NotNaN<f32>)]) {
        // Write Lock!
        let mut node = node.write();

        debug_assert_eq!(UNUSED, node[0]);
        for (i, &(idx, _)) in neighbors.iter().enumerate().take(node.len()) {
            node[i] = idx.into();
        }
    }

    fn connect_nodes(elements: &Elements, node: &RwLock<&mut HnswNode>, i: usize, j: usize, d: NotNaN<f32>) {
        if i == j {
            return;
        }

        // Write Lock!
        let mut node = node.write();

        // Do not insert duplicates
        let j_id: NeighborId = j.into();
        if let Some(free_pos) = node.iter().position(|x| *x == UNUSED || *x == j_id) {
            node[free_pos] = j_id;
        } else {
            let num_neighbors = node.len();
            Self::add_and_limit_neighbors(elements, &mut node, i, &[(j, d)], num_neighbors);
        }
    }

    fn add_and_limit_neighbors(
        elements: &Elements,
        node: &mut HnswNode,
        node_id: usize,
        extra: &[(usize, NotNaN<f32>)],
        num_neighbors: usize,
    ) {
        assert!(num_neighbors <= node.len());

        let element: Element = elements.at(node_id);

        let mut candidates: Vec<_> = iter_neighbors(&node)
            .map(|k| (k, elements.at(k).dist(&element)))
            .collect();

        for &(j, d) in extra {
            candidates.push((j as usize, d));
        }

        candidates.sort_unstable_by_key(|&(_, d)| d);

        let neighbors = Self::select_neighbors(elements, candidates, num_neighbors);

        // set new neighbors and mark last positions as unused
        for (k, n) in neighbors
            .into_iter()
            .map(|(n, _)| n.into())
            .chain(std::iter::repeat(UNUSED))
            .enumerate()
            .take(node.len())
        {
            node[k] = n;
        }
    }
}

impl<'a, Elements, Element> HnswBuilder<'a, Elements, Element>
where
    Elements: 'a + Writeable + At<Output = Element> + Sync + Send + ToOwned + ?Sized,
    Element: 'a + ComparableTo<Element> + Sync + Send,
{
    pub fn save_elements_to_disk(self: &Self, path: &str) -> Result<()> {
        self.get_index().save_elements_to_disk(path)
    }
}

impl<'a, Elements, Element, QElement> HnswBuilder<'a, Elements, Element>
where
    Elements: At<Output = Element> + Sync + Send + ToOwned + ?Sized,
    Elements::Owned: Appendable<Element = QElement> + At<Output = Element> + Sync + Send,
    Element: ComparableTo<Element> + Sync + Send,
{
    pub fn new(config: Config) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Owned(Elements::Owned::new()),
            config: config,
        }
    }

    pub fn append(self: &mut Self, element: QElement) {
        assert!(self.elements.len() + 1 <= <NeighborId>::max_value() as usize);

        self.elements.to_mut().append(element);
    }
}

// Computes a layer multiplier m, s.t. the number of elements in layer i is
// equal to m^i
fn compute_layer_multiplier(num_elements: usize, num_layers: usize) -> f32 {
    (num_elements as f32).powf(1.0 / (num_layers - 1) as f32)
}

pub enum Layers<'a> {
    FixWidth(Vec<FixedWidthSliceVector<'a, NeighborId>>),
    VarWidth(Vec<VariableWidthSliceVector<'a, NeighborId, NeighborId>>),
}

impl<'a> At for VariableWidthSliceVector<'a, NeighborId, NeighborId> {
    type Output = Vec<usize>;

    fn at(self: &Self, index: usize) -> Self::Output {
        iter_neighbors(self.get(index)).collect()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

impl<'a> At for FixedWidthSliceVector<'a, NeighborId> {
    type Output = Vec<usize>;

    fn at(self: &Self, index: usize) -> Self::Output {
        iter_neighbors(self.get(index)).collect()
    }

    fn len(self: &Self) -> usize {
        self.len()
    }
}

pub struct Hnsw<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + ?Sized,
    Element: 'a + ComparableTo<Element>,
{
    layers: Layers<'a>,
    elements: &'a Elements,
}

impl<'a, Elements, Element> Search for Hnsw<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + ?Sized,
    Element: 'a + ComparableTo<Element>,
{
    type Element = Element;
    fn search(&self, element: &Element, num_neighbors: usize, max_search: usize) -> Vec<(usize, f32)> {
        match self.layers {
            Layers::FixWidth(ref layers) => {
                Self::search_internal(layers, &self.elements, element, num_neighbors, max_search)
            }
            Layers::VarWidth(ref layers) => {
                Self::search_internal(layers, &self.elements, element, num_neighbors, max_search)
            }
        }
    }
}

impl<'a, Elements, Element> Hnsw<'a, Elements, Element>
where
    Elements: 'a + At<Output = Element> + ?Sized,
    Element: 'a + ComparableTo<Element>,
{
    pub fn new(layers: Layers<'a>, elements: &'a Elements) -> Self {
        Self { layers, elements }
    }

    pub fn load(buffer: &'a [u8], elements: &'a Elements) -> Self {
        Self {
            layers: io::load_layers(buffer),
            elements: elements,
        }
    }

    pub fn save_index_to_disk(self: &Self, path: &str, compress: bool) -> Result<()> {
        let mut file = File::create(path)?;
        io::save_index_to_disk(&self.layers, &mut file, compress)
    }

    fn search_internal<Layer: At<Output = Vec<usize>>>(
        layers: &[Layer],
        elements: &Elements,
        element: &Element,
        num_neighbors: usize,
        max_search: usize,
    ) -> Vec<(usize, f32)> {
        if let Some((bottom_layer, top_layers)) = layers.split_last() {
            let entrypoint = Self::find_entrypoint(&top_layers, element, elements);

            Self::search_for_neighbors(bottom_layer, entrypoint, elements, element, max_search)
                .into_iter()
                .take(num_neighbors)
                .map(|(i, d)| (i, d.into_inner()))
                .collect()
        } else {
            vec![]
        }
    }

    fn find_entrypoint<Layer: At<Output = Vec<usize>>>(
        layers: &[Layer],
        element: &Element,
        elements: &Elements,
    ) -> usize {
        let mut entrypoint = 0;
        for layer in layers {
            let res = Self::search_for_neighbors(layer, entrypoint, elements, &element, 1);

            entrypoint = res[0].0;
        }

        entrypoint
    }

    fn search_for_neighbors<Layer: At<Output = Vec<usize>>>(
        layer: &Layer,
        entrypoint: usize,
        elements: &Elements,
        goal: &Element,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {
        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();

        let num_neighbors = layer.at(0).len();
        let mut visited = hashbrown::HashSet::with_capacity(max_search * num_neighbors);

        let distance = elements.at(entrypoint).dist(&goal);

        pq.push(RevOrd((distance, entrypoint)));

        visited.insert(entrypoint);

        while let Some(RevOrd((d, idx))) = pq.pop() {
            if res.is_full() && d > res.peek().unwrap().0 {
                break;
            }

            res.push((d, idx));

            for neighbor_idx in layer.at(idx) {
                if visited.insert(neighbor_idx) {
                    let distance = elements.at(neighbor_idx).dist(&goal);

                    if !res.is_full() || distance < res.peek().unwrap().0 {
                        pq.push(RevOrd((distance, neighbor_idx)));
                    }
                }
            }
        }

        res.heap
            .into_sorted_vec()
            .into_iter()
            .map(|(d, idx)| (idx, d))
            .collect()
    }

    pub fn len(self: &Self) -> usize {
        self.layer_len(self.num_layers() - 1)
    }

    pub fn num_layers(self: &Self) -> usize {
        match self.layers {
            Layers::FixWidth(ref layers) => layers.len(),
            Layers::VarWidth(ref layers) => layers.len(),
        }
    }

    pub fn get_element(self: &Self, index: usize) -> Element {
        self.elements.at(index)
    }

    pub fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize> {
        match self.layers {
            Layers::FixWidth(ref layers) => layers[layer].at(index),
            Layers::VarWidth(ref layers) => layers[layer].at(index),
        }
    }

    pub fn layer_len(self: &Self, layer: usize) -> usize {
        match self.layers {
            Layers::FixWidth(ref layers) => layers[layer].len(),
            Layers::VarWidth(ref layers) => layers[layer].len(),
        }
    }

    pub fn count_neighbors(self: &Self, layer: usize, begin: usize, end: usize) -> usize {
        match self.layers {
            Layers::FixWidth(ref layers) => layers[layer]
                .par_iter()
                .skip(begin)
                .take(end - begin)
                .map(|node| iter_neighbors(node).count())
                .sum(),
            // todo: this can be done in constant time for Layers::VarWidth
            Layers::VarWidth(ref layers) => layers[layer]
                .iter()
                .skip(begin)
                .take(end - begin)
                .map(|node| iter_neighbors(node).count())
                .sum(),
        }
    }
}

impl<'a, Elements, Element> Hnsw<'a, Elements, Element>
where
    Elements: Writeable + At<Output = Element> + ToOwned + ?Sized,
    Element: ComparableTo<Element>,
{
    pub fn save_elements_to_disk(self: &Self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut file = BufWriter::new(file);

        self.elements.write(&mut file)
    }
}

struct MaxSizeHeap<T> {
    heap: BinaryHeap<T>,
    max_size: usize,
}

impl<T: Ord> MaxSizeHeap<T> {
    pub fn new(max_size: usize) -> Self {
        MaxSizeHeap {
            heap: BinaryHeap::with_capacity(max_size),
            max_size: max_size,
        }
    }

    pub fn push(self: &mut Self, element: T) -> bool {
        if !self.is_full() {
            self.heap.push(element);
            true
        } else if element < *self.heap.peek().unwrap() {
            if self.heap.len() >= self.max_size {
                self.heap.pop();
            }

            self.heap.push(element);
            true
        } else {
            false
        }
    }

    pub fn is_full(self: &Self) -> bool {
        self.heap.len() >= self.max_size
    }

    pub fn peek(self: &Self) -> Option<&T> {
        self.heap.peek()
    }
}
