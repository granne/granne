use fnv::FnvHashSet;
use ordered_float::NotNaN;
use revord::RevOrd;
use std::collections::BinaryHeap;
use std::cmp;
use pbr::ProgressBar;

use slice_vector::{FixedWidthSliceVector, SliceVector};
use serde_json;

// Write and read
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Result};

use byteorder::{LittleEndian, ReadBytesExt};

// Threading
use rayon::prelude::*;
use parking_lot::{Mutex, RwLock};

use time::PreciseTime;

use std::borrow::Cow;

use crate::types::ComparableTo;
use crate::types;
use crate::file_io;


#[cfg(test)]
mod tests;
mod sharded_hnsw;
mod bloomfilter;
mod neighborid;

pub use self::sharded_hnsw::ShardedHnsw;

use self::neighborid::NeighborId;
const MAX_NEIGHBORS: usize = 32;
const EXTRA_NEIGHBORS_AT_BUILD_TIME: usize = 6;
const UNUSED: NeighborId = NeighborId([<u8>::max_value(); 5]);
const METADATA_LEN: usize = 1024;
const LIBRARY_STR: &str = "granne";
const SERIALIZATION_VERSION: usize = 0;

type HnswNode = [NeighborId];

fn iter_neighbors<'b>(node: &'b HnswNode) -> impl 'b + Iterator<Item=usize> {
    node.iter()
        .take_while(|&&n| n != UNUSED)
        .map(|&n| n.into())
}

#[derive(Clone)]
pub struct Config {
    pub num_layers: usize,
    pub num_neighbors: usize,
    pub max_search: usize,
    pub show_progress: bool,
}

impl Config {
    pub fn default() -> Self {
        Self {
            num_layers: 7,
            num_neighbors: 20,
            max_search: 200,
            show_progress: true,
        }
    }
}

pub struct HnswBuilder<'a, Elements, Element>
    where Elements: 'a + At<Output=Element> + Sync + Send + ToOwned + ?Sized,
          Element: 'a + ComparableTo<Element> + Sync + Send
{
    layers: Vec<FixedWidthSliceVector<'static, NeighborId>>,
    elements: Cow<'a, Elements>,
    config: Config
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
    type Output=T;

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

pub struct Hnsw<'a, Elements, Element>
    where Elements: 'a + At<Output=Element> + ?Sized,
          Element: 'a + ComparableTo<Element>
{
    layers: Vec<FixedWidthSliceVector<'a, NeighborId>>,
    elements: &'a Elements,
}



impl<'a, Elements, Element> HnswBuilder<'a, Elements, Element>
    where Elements: 'a + At<Output=Element> + Sync + Send + ToOwned + ?Sized,
          Element: 'a + ComparableTo<Element> + Sync + Send + Clone
{
    pub fn with_borrowed_elements(config: Config, elements: &'a Elements) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Borrowed(elements),
            config: config
        }
    }


    pub fn with_owned_elements(config: Config, elements: Elements::Owned) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Owned(elements),
            config: config
        }
    }


    pub fn read_index_with_owned_elements<I: Read>(config: Config, index_reader: &mut I, elements: Elements::Owned) -> Result<Self>
    {
        let layers = read_layers(index_reader)?;

        let elements: Cow<Elements> = Cow::Owned(elements);

        if let Some(ref last_layer) = layers.last() {
            assert!(last_layer.len() <= elements.len());
        }

        Ok(Self {
            layers: layers,
            elements: elements,
            config: config
        })
    }


    pub fn read_index_with_borrowed_elements<I: Read>(config: Config, index_reader: &mut I, elements: &'a Elements) -> Result<Self>
    {
        let layers = read_layers(index_reader)?;

        if let Some(ref last_layer) = layers.last() {
            assert!(last_layer.len() <= elements.len());
        }

        Ok(Self {
            layers: layers,
            elements: Cow::Borrowed(elements),
            config: config
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


    pub fn save_index_to_disk(self: &Self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut file = BufWriter::new(file);
        self.write(&mut file)
    }


    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        let mut metadata = String::with_capacity(METADATA_LEN);
        metadata.push_str(LIBRARY_STR);

        metadata.push_str(&serde_json::to_string(
            &json!({
                "version": SERIALIZATION_VERSION,
                "num_neighbors": self.config.num_neighbors,
                "num_extra_neighbors_at_build_time": EXTRA_NEIGHBORS_AT_BUILD_TIME,
                "num_elements": self.len(),
                "num_indexed_elements": self.indexed_elements(),
                "num_layers": self.layers.len(),
                "layer_counts": self.layers.iter().map(|layer| layer.len()).collect::<Vec<_>>(),
            })
        ).expect("Could not create metadata json"));

        let mut metadata = metadata.into_bytes();
        assert!(metadata.len() <= METADATA_LEN);
        metadata.resize(METADATA_LEN, ' ' as u8);

        buffer.write_all(metadata.as_slice())?;

        // write graph
        for layer in &self.layers {
            if EXTRA_NEIGHBORS_AT_BUILD_TIME > 0 {
                for node in layer.iter() {
                    file_io::write(&node[..self.config.num_neighbors], buffer)?;
                }
            } else {
                layer.write(buffer)?;
            }
        }

        Ok(())
    }


    pub fn get_index<'b>(self: &'b Self) -> Hnsw<'b, Elements, Element> {
        Hnsw {
            layers: self.layers.iter().map(|layer| layer.borrow()).collect(),
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

        assert!(num_elements >= self.layers.last().map_or(0, |layer| layer.len()),
                "Cannot index fewer elements than already in index.");
        assert!(num_elements <= self.elements.len(),
                "Cannot index more elements than exist.");

        // fresh index build => initialize first layer
        if self.layers.is_empty() {
            let mut layer =
                FixedWidthSliceVector::new(self.config.num_neighbors + EXTRA_NEIGHBORS_AT_BUILD_TIME);
            layer.resize(1, UNUSED);
            self.layers.push(layer);
        }
        else {
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
        let layer_multiplier =
            compute_layer_multiplier(self.elements.len(), self.config.num_layers);

        let layer = self.layers.len()-1;
        let ideal_num_elements_in_layer = cmp::min(layer_multiplier.pow(layer as u32), self.elements.len());
        let mut num_elements_in_layer = cmp::min(max_num_elements, ideal_num_elements_in_layer);

        // if last layer index all elements
        if layer == self.config.num_layers - 1 {
            num_elements_in_layer = max_num_elements;
        }

        let mut layer = self.layers.pop().unwrap();

        let additional = ideal_num_elements_in_layer - layer.len();
        layer.reserve_exact(additional);

        Self::index_elements(&self.config, &self.elements, num_elements_in_layer, &self.get_index(), &mut layer);

        self.layers.push(layer);
    }


    fn index_elements(
        config: &Config,
        elements: &Elements,
        num_elements: usize,
        prev_layers: &Hnsw<Elements,Element>,
        layer: &mut FixedWidthSliceVector<'static, NeighborId>,
    ) {
        assert!(layer.len() <= num_elements);

        let already_indexed = layer.len();

        layer.resize(num_elements, UNUSED);
        // set up progress bar
        let step_size = cmp::max(100, num_elements / 400);
        let (progress_bar, start_time) = {
            if config.show_progress {
                let mut progress_bar = ProgressBar::new(elements.len() as u64);
                let info_text = format!("Layer {}: ", prev_layers.layers.len());
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
            let layer: Vec<RwLock<&mut HnswNode>> =
                layer.iter_mut().map(|node| RwLock::new(node)).collect();

            // index elements, skipping already indexed
            layer
                .par_iter()
                .enumerate()
                .skip(already_indexed)
                .for_each(|(idx, _)| {
                    Self::index_element(config, elements, prev_layers, &layer, idx);

                    // This only shows approximate progress because of par_iter
                    if idx % step_size == 0 {
                        if let Some(ref progress_bar) = progress_bar {
                            progress_bar.lock().add(step_size as u64);
                        }
                    }
                });
        }

        if let Some(progress_bar) = progress_bar {
            progress_bar.lock().set(layer.len() as u64);

            if let Some(start_time) = start_time {
                let end_time = PreciseTime::now();
                println!("Time: {} s", start_time.to(end_time).num_seconds());
            }
        }

        // make sure all nodes have at most config.num_neighbors neighbors
        if EXTRA_NEIGHBORS_AT_BUILD_TIME > 0 {
            layer.par_iter_mut().enumerate().for_each(|(i, node)| {
                if node.len() > config.num_neighbors {
                    Self::add_and_limit_neighbors(elements, node, i, &[], config.num_neighbors);
                }
            });
        }
    }


    fn index_element(
        config: &Config,
        elements: &Elements,
        prev_layers: &Hnsw<Elements, Element>,
        layer: &[RwLock<&mut HnswNode>],
        idx: usize,
    ) {

        let element: Element = elements.at(idx);

        // do not index elements that are zero (multiply by 100 as safety margin)
        if element.dist(&element) > NotNaN::new(100.0).unwrap() * Element::eps() {
            return;
        }

        let (entrypoint, _) = prev_layers.search(&element, 1, 1)[0];

        let candidates = Self::search_for_neighbors_index(
            elements,
            layer,
            entrypoint,
            &element,
            config.max_search,
        );

        let neighbors = Self::select_neighbors(elements, candidates, config.num_neighbors);

        // if the current element is a duplicate of too many of its potential neighbors, do not connect it to the graph,
        // this effectively creates a dead node
        if let Some((_, d)) = neighbors.get(config.num_neighbors / 2) {
            if *d < Element::eps() {
                return;
            }
        }

        Self::initialize_node(&layer[idx], &neighbors[..]);

        for (neighbor, d) in neighbors {
            Self::connect_nodes(elements, &layer[neighbor], neighbor, idx, d);
        }
    }


    // Similar to Hnsw::search_for_neighbors but with RwLocks for
    // parallel indexing (and bloomfilter for visited set)
    fn search_for_neighbors_index(
        elements: &Elements,
        layer: &[RwLock<&mut HnswNode>],
        entrypoint: usize,
        goal: &Element,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {

        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();

        // A bloomfilter is fine since we mostly want to avoid revisiting nodes
        // and it's faster than FnvHashSet.
        let mut visited = bloomfilter::BloomFilter::new(max_search * MAX_NEIGHBORS, 0.01);

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
            if neighbors.iter().all(
                |&(_, _, ref neighbor)| d <= neighbor.dist(&element) + Element::eps()
            ) {
                neighbors.push((j, d, element));
            } else {
                pruned.push((j, d));
            }
        }

        let neighbors: Vec<_> = neighbors.into_iter().map(|(j, d, _)| (j,d)).collect();

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


    fn connect_nodes(
        elements: &Elements,
        node: &RwLock<&mut HnswNode>,
        i: usize,
        j: usize,
        d: NotNaN<f32>,
    ) {
        // Write Lock!
        let mut node = node.write();

        if let Some(free_pos) = node.iter().position(|x| *x == UNUSED) {
            node[free_pos] = j.into();
        } else {
            let num_neighbors = node.len() - EXTRA_NEIGHBORS_AT_BUILD_TIME;
            Self::add_and_limit_neighbors(elements, &mut node, i, &[(j, d)], num_neighbors);
        }
    }


    fn add_and_limit_neighbors(elements: &Elements, node: &mut HnswNode, node_id: usize, extra: &[(usize, NotNaN<f32>)], num_neighbors: usize) {
        assert!(num_neighbors <= node.len());

        let element: Element = elements.at(node_id);

        let mut candidates: Vec<_> = iter_neighbors(&node)
            .map(|k| (k, elements.at(k).dist(&element)))
            .collect();

        for &(j, d) in extra {
            candidates.push((j as usize, d));
        }

        candidates.sort_unstable_by_key(|&(_, d)| d);

        let neighbors =
            Self::select_neighbors(elements, candidates, num_neighbors);

        // set new neighbors and mark last positions as unused
        for (k, n) in neighbors.into_iter()
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
    where Elements: 'a + Writeable + At<Output=Element> + Sync + Send + ToOwned + ?Sized,
          Element: 'a + ComparableTo<Element> + Sync + Send
{
    pub fn save_elements_to_disk(self: &Self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut file = BufWriter::new(file);

        self.elements.write(&mut file)
    }
}


// Methods only implemented for AngularVectors
impl<'a, T> HnswBuilder<'a, types::AngularVectorsT<'static, T>, types::AngularVectorT<'static, T>> where
    T: Copy + Sync + Send,
    types::AngularVectorT<'static, T>: ComparableTo<types::AngularVectorT<'static, T>>
{
    pub fn new(dimension: usize, config: Config) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::Owned(types::AngularVectorsT::new(dimension)),
            config: config
        }
    }


    pub fn add(self: &mut Self, elements: types::AngularVectorsT<'static, T>) {
        assert!(self.elements.len() + (elements.len() / self.elements.dim) <= <NeighborId>::max_value() as usize);

        if self.elements.len() == 0 {
            self.elements = Cow::Owned(elements);
        } else {
            self.elements.to_mut().extend(elements);
        }
    }
}


// Computes a layer multiplier m, s.t. the number of elements in layer i is
// equal to m^i
fn compute_layer_multiplier(num_elements: usize, num_layers: usize) -> usize {
    (num_elements as f32)
        .powf(1.0 / (num_layers - 1) as f32)
        .ceil() as usize
}


impl<'a, Elements, Element> Hnsw<'a, Elements, Element>
    where Elements: 'a + At<Output=Element> + ?Sized,
          Element: 'a + ComparableTo<Element>
{
    pub fn load(buffer: &'a [u8], elements: &'a Elements) -> Self {
        let (num_neighbors, layer_counts) = read_metadata(buffer).expect("Could not read metadata");

        // load graph
        let mut layers = Vec::new();
        let node_size = num_neighbors * std::mem::size_of::<NeighborId>();

        let mut start = METADATA_LEN;
        for count in layer_counts {
            let end = start + count * node_size;
            let layer = &buffer[start..end];
            layers.push(FixedWidthSliceVector::load(layer, num_neighbors));
            start = end;
        }

        Self {
            layers: layers,
            elements: elements,
        }

    }


    pub fn search(
        &self,
        element: &Element,
        num_neighbors: usize,
        max_search: usize,
    ) -> Vec<(usize, f32)> {

        let (bottom_layer, top_layers) = self.layers.split_last().unwrap();

        let entrypoint = Self::find_entrypoint(&top_layers, element, &self.elements);

        Self::search_for_neighbors(
            &bottom_layer,
            entrypoint,
            &self.elements,
            element,
            max_search,
        ).into_iter()
            .take(num_neighbors)
            .map(|(i, d)| (i, d.into_inner()))
            .collect()
    }


    fn find_entrypoint(
        layers: &[FixedWidthSliceVector<'a, NeighborId>],
        element: &Element,
        elements: &Elements) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res =
                Self::search_for_neighbors(&layer, entrypoint, elements, &element, 1);

            entrypoint = res[0].0;
        }

        entrypoint
    }


    fn search_for_neighbors(
        layer: &FixedWidthSliceVector<'a, NeighborId>,
        entrypoint: usize,
        elements: &Elements,
        goal: &Element,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {

        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited =
            FnvHashSet::with_capacity_and_hasher(max_search * MAX_NEIGHBORS, Default::default());

        let distance = elements.at(entrypoint).dist(&goal);

        pq.push(RevOrd((distance, entrypoint)));

        visited.insert(entrypoint);

        while let Some(RevOrd((d, idx))) = pq.pop() {
            if res.is_full() && d > res.peek().unwrap().0 {
                break;
            }

            res.push((d, idx));

            let node = layer.get(idx);

            for neighbor_idx in iter_neighbors(&node) {
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
        self.elements.len()
    }

    pub fn get_element(self: &Self, index: usize) -> Element {
        self.elements.at(index)
    }

    pub fn get_neighbors(self: &Self, index: usize, layer: usize) -> Vec<usize> {
        iter_neighbors(self.layers[layer].get(index)).collect()
    }

}


fn read_metadata<I: Read>(index_reader: I) -> Result<(usize, Vec<usize>)> {
    let mut index_reader = index_reader.take(METADATA_LEN as u64);

    // Check if the file is a current or old version of an granne index
    let mut lib_str = Vec::new();
    index_reader.by_ref().take(LIBRARY_STR.len() as u64).read_to_end(&mut lib_str)?;
    if String::from_utf8(lib_str).unwrap_or("".to_string()) == LIBRARY_STR {
        // the current version stores metadata as json
        let metadata: serde_json::Value = serde_json::from_reader(index_reader).expect("Could not read metadata");

        let num_neighbors: usize =
            serde_json::from_value(metadata["num_neighbors"].clone()).expect("Could not read num_neighbors");
        let num_layers: usize =
            serde_json::from_value(metadata["num_layers"].clone()).expect("Could not read num_layers");
        let layer_counts: Vec<usize> =
            serde_json::from_value(metadata["layer_counts"].clone()).expect("Could not read layer_counts");
        assert_eq!(num_layers, layer_counts.len());

        Ok((num_neighbors, layer_counts))
    } else {
        // Read legacy index
        // First 8 bytes are num_nodes. We can ignore these.
        // Note that "granne".len() bytes were already read.
        const LIBRARY_STR_LEN: usize = 6;
        assert_eq!(LIBRARY_STR_LEN, LIBRARY_STR.len());
        const BYTES_LEFT_FOR_NUM_NODES: usize = std::mem::size_of::<usize>() - LIBRARY_STR_LEN;

        index_reader.read_exact(&mut [0; BYTES_LEFT_FOR_NUM_NODES])?;

        let num_layers = index_reader.read_u64::<LittleEndian>()? as usize;

        let mut layer_counts = Vec::new();
        for _ in 0..num_layers {
            layer_counts.push(index_reader.read_u64::<LittleEndian>()? as usize);
        }

        let num_neighbors = 20;

        Ok((num_neighbors, layer_counts))
    }
}


fn read_layers<I: Read>(index_reader: I) -> Result<Vec<FixedWidthSliceVector<'static, NeighborId>>> {
    use std::mem::size_of;

    let mut index_reader = BufReader::new(index_reader);

    let (num_neighbors, layer_counts) = read_metadata(index_reader.by_ref())?;

    // read graph
    let mut layers = Vec::new();
    let node_size = num_neighbors * size_of::<NeighborId>();

    for count in layer_counts {
        let mut layer_reader = index_reader.by_ref().take((node_size * count) as u64);

        let layer = if EXTRA_NEIGHBORS_AT_BUILD_TIME > 0 {
            let num_neighbors_at_build_time = num_neighbors + EXTRA_NEIGHBORS_AT_BUILD_TIME;

            let mut vec = FixedWidthSliceVector::with_capacity(num_neighbors_at_build_time, count);

            let mut buffer = Vec::new();
            buffer.resize(num_neighbors_at_build_time * size_of::<NeighborId>(), 0);

            while let Ok(()) = layer_reader.read_exact(&mut buffer[..node_size]) {
                vec.push(file_io::load(&buffer));
            }

            vec.par_iter_mut().for_each(|node| {
                for i in num_neighbors..num_neighbors_at_build_time {
                    node[i] = UNUSED;
                }
            });

            vec

        } else {
            FixedWidthSliceVector::read(layer_reader, num_neighbors)?
        };

        assert_eq!(count, layer.len());

        layers.push(layer);
    }

    Ok(layers)
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

    pub fn push(self: &mut Self, element: T) {
        if !self.is_full() {
            self.heap.push(element);

        } else if element < *self.heap.peek().unwrap() {
            if self.heap.len() >= self.max_size {
                self.heap.pop();
            }

            self.heap.push(element);
        }
    }

    pub fn is_full(self: &Self) -> bool {
        self.heap.len() >= self.max_size
    }

    pub fn peek(self: &Self) -> Option<&T> {
        self.heap.peek()
    }
}
