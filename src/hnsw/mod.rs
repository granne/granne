use arrayvec::ArrayVec;
use fnv::FnvHashSet;
use ordered_float::NotNaN;
use revord::RevOrd;
use std::collections::BinaryHeap;
use std::cmp;
use types::ComparableTo;
use pbr::ProgressBar;

// Write and read
use std::fs::File;
use std::io::{Read, Write, Result};

// Threading
use rayon::prelude::*;
use std::sync::{Mutex, RwLock};

use time::PreciseTime;

use std::marker::PhantomData;

use std::borrow::Cow;

use file_io;

mod tests;
mod generic;

pub use self::generic::{SearchIndex, IndexBuilder, boxed_index, boxed_index_builder};

const MAX_NEIGHBORS: usize = 20;
type NeighborType = u32;

#[repr(C)]
#[derive(Clone, Default, Debug)]
struct HnswNode {
    neighbors: ArrayVec<[NeighborType; MAX_NEIGHBORS]>,
}

#[derive(Clone)]
pub struct Config {
    pub num_layers: usize,
    pub max_search: usize,
    pub show_progress: bool,
}

pub struct HnswBuilder<'a, T: 'a + ComparableTo<T> + Clone + Sync + Send> {
    layers: Vec<Vec<HnswNode>>,
    elements: Cow<'a, [T]>,
    config: Config,
}


pub struct Hnsw<'a, T: ComparableTo<E> + 'a, E> {
    layers: Vec<&'a [HnswNode]>,
    elements: &'a [T],
    phantom: PhantomData<E>,
}


impl<'a, T: 'a + ComparableTo<T> + Sync + Send + Clone> HnswBuilder<'a, T> {
    pub fn new(config: Config) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::from(Vec::new()),
            config: config,
        }
    }

    pub fn with_elements(config: Config, elements: &'a [T]) -> Self {
        HnswBuilder {
            layers: Vec::new(),
            elements: Cow::from(elements),
            config: config,
        }
    }

    pub fn save_elements_to_disk(self: &Self, path: &str) -> Result<()> {

        file_io::save_to_disk(&self.elements[..], &path)
    }

    pub fn save_index_to_disk(self: &Self, path: &str) -> Result<()> {

        let mut file = File::create(path)?;

        self.write(&mut file)
    }


    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        // write metadata
        let num_nodes = self.layers.iter().map(|layer| layer.len()).sum();
        let num_layers = self.layers.len();
        let layer_counts = self.layers.iter().map(|layer| layer.len());

        let mut usize_data = vec![num_nodes, num_layers];
        usize_data.extend(layer_counts);

        let data = unsafe {
            ::std::slice::from_raw_parts(
                usize_data.as_ptr() as *const u8,
                usize_data.len() * ::std::mem::size_of::<usize>(),
            )
        };

        buffer.write_all(data)?;

        // write graph
        for layer in &self.layers {

            let data = unsafe {
                ::std::slice::from_raw_parts(
                    layer.as_ptr() as *const u8,
                    layer.len() * ::std::mem::size_of::<HnswNode>(),
                )
            };

            buffer.write_all(data)?;
        }

        Ok(())
    }

    pub fn read<I: Read, E: Read>(index_reader: &mut I, element_reader: &mut E) -> Result<Self> {
        use std::mem::size_of;

        // read metadata
        const BUFFER_SIZE: usize = 512;
        let mut buffer = [0u8; BUFFER_SIZE];
        index_reader.read_exact(&mut buffer[..2 * size_of::<usize>()])?;

        let num_nodes: usize = unsafe { *(&buffer[0] as *const u8 as *const usize) };

        let num_layers = unsafe { *(&buffer[1 * size_of::<usize>()] as *const u8 as *const usize) };

        index_reader.read_exact(&mut buffer[..num_layers * size_of::<usize>()])?;

        let mut layer_counts = Vec::new();

        for i in 0..num_layers {
            let count: usize =
                unsafe { *(&buffer[i * size_of::<usize>()] as *const u8 as *const usize) };

            layer_counts.push(count);
        }

        assert_eq!(num_nodes, layer_counts.iter().sum::<usize>());

        // read graph
        let mut layers = Vec::new();

        for count in layer_counts {
            layers.push(Vec::<HnswNode>::with_capacity(count));

            for _ in 0..count {
                index_reader.read_exact(&mut buffer[..size_of::<HnswNode>()])?;

                layers.last_mut().unwrap().push(unsafe {
                    (*(&buffer[0] as *const u8 as *const HnswNode)).clone()
                })
            }
        }

        // read elements
        let num_elements = layers.last().unwrap().len();
        let mut elements: Vec<T> = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            element_reader.read_exact(&mut buffer[..size_of::<T>()])?;

            elements.push(unsafe { (*(&buffer[0] as *const u8 as *const T)).clone() })
        }

        let config = Config {
            num_layers: layers.len(),
            max_search: 200,
            show_progress: true,
        };

        Ok(Self {
            layers: layers,
            elements: Cow::from(elements),
            config: config,
        })
    }


    pub fn get_index<'b>(self: &'b Self) -> Hnsw<'b, T, T> {
        Hnsw {
            layers: self.layers.iter().map(|layer| &layer[..]).collect(),
            elements: &self.elements[..],
            phantom: PhantomData,
        }
    }


    pub fn from_index(config: Config, index: &Hnsw<T, T>) -> Self {
        let mut builder = Self::new(config);

        builder.elements = Cow::from(index.elements.to_vec());
        builder.layers = index.layers.iter().map(|layer| layer.to_vec()).collect();

        builder
    }


    pub fn build_index(&mut self) {
        // unbuilt index
        if self.layers.is_empty() {
            self.layers.push(vec![HnswNode::default()]);

            let layer_multiplier =
                compute_layer_multiplier(self.elements.len(), self.config.num_layers);

            let mut num_elements_in_layer = 1;
            for layer in 1..self.config.num_layers {
                num_elements_in_layer *= layer_multiplier;

                if num_elements_in_layer > self.elements.len() ||
                    layer == self.config.num_layers - 1
                {
                    num_elements_in_layer = self.elements.len();
                }

                // copy layer above
                let mut new_layer = Vec::with_capacity(num_elements_in_layer);
                new_layer.extend_from_slice(self.layers.last().unwrap());

                Self::insert_elements(
                    &self.config,
                    &self.elements[..num_elements_in_layer],
                    &self.get_index(),
                    &mut new_layer,
                );

                self.layers.push(new_layer);

                if num_elements_in_layer == self.elements.len() {
                    break;
                }
            }
        }
        // inserting recently added elements only
        else {
            let mut layer = self.layers.pop().unwrap();

            Self::insert_elements(&self.config, &self.elements, &self.get_index(), &mut layer);

            self.layers.push(layer);
        }
    }


    pub fn add(self: &mut Self, elements: Vec<T>) {
        assert!(self.elements.len() + elements.len() <= <NeighborType>::max_value() as usize);

        if self.elements.is_empty() {
            self.elements = Cow::from(elements);
        } else {
            self.elements.to_mut().extend_from_slice(elements.as_slice());
        }
    }

    fn insert_elements(
        config: &Config,
        elements: &[T],
        prev_layers: &Hnsw<T,T>,
        layer: &mut Vec<HnswNode>,
    ) {

        assert!(layer.len() <= elements.len());

        let already_inserted = layer.len();

        layer.resize(elements.len(), HnswNode::default());

        assert_eq!(layer.len(), layer.capacity());

        // create RwLocks for underlying nodes
        let layer: Vec<RwLock<&mut HnswNode>> =
            layer.iter_mut().map(|node| RwLock::new(node)).collect();

        assert_eq!(layer.len(), layer.capacity());

        // set up progress bar
        let step_size = cmp::max(100, elements.len() / 400);
        let (progress_bar, start_time) = {
            if config.show_progress {
                let mut progress_bar = ProgressBar::new(elements.len() as u64);
                let info_text = format!("Layer {}: ", prev_layers.layers.len());
                progress_bar.message(&info_text);
                progress_bar.set((step_size * (already_inserted / step_size)) as u64);

                (Some(Mutex::new(progress_bar)), Some(PreciseTime::now()))

            } else {
                (None, None)
            }
        };

        // insert elements, skipping already inserted
        elements
            .par_iter()
            .enumerate()
            .skip(already_inserted)
            .for_each(|(idx, _)| {
                Self::insert_element(config, elements, prev_layers, &layer, idx);

                // This only shows approximate progress because of par_iter
                if idx % step_size == 0 {
                    if let Some(ref progress_bar) = progress_bar {
                        progress_bar.lock().unwrap().add(step_size as u64);
                    }
                }
            });

        if let Some(progress_bar) = progress_bar {
            progress_bar.lock().unwrap().finish_println("");

            if let Some(start_time) = start_time {
                let end_time = PreciseTime::now();
                println!("Time: {} s", start_time.to(end_time).num_seconds());
            }
        }
    }


    fn insert_element(
        config: &Config,
        elements: &[T],
        prev_layers: &Hnsw<T,T>,
        layer: &Vec<RwLock<&mut HnswNode>>,
        idx: usize,
    ) {

        let element = &elements[idx];

        let (entrypoint, _) = prev_layers.search(element, 1, 1)[0];

        let neighbors = Self::search_for_neighbors_index(
            elements,
            &layer[..],
            entrypoint,
            element,
            config.max_search,
        );

        let neighbors = Self::select_neighbors(elements, neighbors, MAX_NEIGHBORS);

        Self::initialize_node(&layer[idx], &neighbors[..]);

        for (neighbor, d) in neighbors {
            Self::connect_nodes(elements, &layer[neighbor], neighbor, idx, d);
        }
    }


    // Similar to Hnsw::search_for_neighbors but with RwLocks for
    // parallel insertion
    fn search_for_neighbors_index(
        elements: &[T],
        layer: &[RwLock<&mut HnswNode>],
        entrypoint: usize,
        goal: &T,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {

        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited =
            FnvHashSet::with_capacity_and_hasher(max_search * MAX_NEIGHBORS, Default::default());

        pq.push(RevOrd((elements[entrypoint].dist(&goal), entrypoint)));

        visited.insert(entrypoint);

        while let Some(RevOrd((d, idx))) = pq.pop() {
            if res.is_full() && d > res.peek().unwrap().0 {
                break;
            }

            res.push((d, idx));

            let node = layer[idx].read().unwrap();

            for neighbor_idx in node.neighbors.iter().map(|&n| n as usize) {
                if visited.insert(neighbor_idx) {
                    let distance = elements[neighbor_idx].dist(&goal);

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
        elements: &[T],
        candidates: Vec<(usize, NotNaN<f32>)>,
        max_neighbors: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {

        if candidates.len() <= max_neighbors {
            return candidates;
        }

        let mut neighbors = Vec::new();
        let mut pruned = Vec::new();

        // candidates are sorted on distance from idx
        for (j, d) in candidates.into_iter() {
            if neighbors.len() >= max_neighbors {
                break;
            }

            // add j to neighbors if j is closer to idx,
            // than to all previously added neighbors
            if neighbors.iter().all(
                |&(k, _)| d < elements[j].dist(&elements[k]),
            )
            {
                neighbors.push((j, d));
            } else {
                pruned.push((j, d));
            }
        }

        let remaining = max_neighbors - neighbors.len();
        neighbors.extend(pruned.into_iter().take(remaining));

        neighbors
    }


    fn initialize_node(node: &RwLock<&mut HnswNode>, neighbors: &[(usize, NotNaN<f32>)]) {
        // Write Lock!
        let mut node = node.write().unwrap();

        debug_assert!(node.neighbors.len() == 0);
        let num_to_add = node.neighbors.capacity() - node.neighbors.len();

        for &(idx, _) in neighbors.iter().take(num_to_add) {
            node.neighbors.push(idx as NeighborType);
        }
    }


    fn connect_nodes(
        elements: &[T],
        node: &RwLock<&mut HnswNode>,
        i: usize,
        j: usize,
        d: NotNaN<f32>,
    ) {
        // Write Lock!
        let mut node = node.write().unwrap();

        if node.neighbors.len() < MAX_NEIGHBORS {
            node.neighbors.push(j as NeighborType);
        } else {

            let mut candidates: Vec<_> = node.neighbors
                .iter()
                .map(|&k| (k as usize, elements[i].dist(&elements[k as usize])))
                .collect();

            candidates.push((j as usize, d));
            candidates.sort_unstable_by_key(|&(_, d)| d);

            let neighbors = Self::select_neighbors(elements, candidates, MAX_NEIGHBORS);

            for (k, (n, _)) in neighbors.into_iter().enumerate() {
                node.neighbors[k] = n as NeighborType;
            }
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


impl<'a, T: ComparableTo<E> + 'a, E> Hnsw<'a, T, E> {
    pub fn load(buffer: &'a [u8], elements: &'a [T]) -> Self {

        let offset = 0 * ::std::mem::size_of::<usize>();
        let num_nodes = &buffer[offset] as *const u8 as *const usize;

        let offset = 1 * ::std::mem::size_of::<usize>();
        let num_layers = &buffer[offset] as *const u8 as *const usize;

        let offset = 2 * ::std::mem::size_of::<usize>();

        let layer_counts: &[usize] = unsafe {
            ::std::slice::from_raw_parts(&buffer[offset] as *const u8 as *const usize, *num_layers)
        };

        let offset = (2 + layer_counts.len()) * ::std::mem::size_of::<usize>();

        let nodes: &[HnswNode] = unsafe {
            ::std::slice::from_raw_parts(
                &buffer[offset] as *const u8 as *const HnswNode,
                *num_nodes,
            )
        };

        assert_eq!(nodes.len(), layer_counts.iter().sum::<usize>());

        let mut layers = Vec::new();

        let mut start = 0;
        for &layer_count in layer_counts {
            let end = start + layer_count;
            let layer = &nodes[start..end];
            layers.push(layer);
            start = end;
        }

        let offset = offset + nodes.len() * ::std::mem::size_of::<HnswNode>();

        assert_eq!(buffer.len(), offset);
        assert_eq!(layers.last().unwrap().len(), elements.len());

        Self {
            layers: layers,
            elements: elements,
            phantom: PhantomData,
        }
    }


    pub fn search(
        &self,
        element: &E,
        num_neighbors: usize,
        max_search: usize,
    ) -> Vec<(usize, f32)> {

        let (bottom_layer, top_layers) = self.layers.split_last().unwrap();

        let entrypoint = Self::find_entrypoint(&top_layers, element, &self.elements, max_search);

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
        layers: &[&[HnswNode]],
        element: &E,
        elements: &[T],
        max_search: usize,
    ) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res =
                Self::search_for_neighbors(&layer, entrypoint, &elements, &element, 1);

            entrypoint = res[0].0;
        }

        entrypoint
    }


    fn search_for_neighbors(
        layer: &[HnswNode],
        entrypoint: usize,
        elements: &[T],
        goal: &E,
        max_search: usize,
    ) -> Vec<(usize, NotNaN<f32>)> {

        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> = MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited =
            FnvHashSet::with_capacity_and_hasher(max_search * MAX_NEIGHBORS, Default::default());

        pq.push(RevOrd((elements[entrypoint].dist(&goal), entrypoint)));

        visited.insert(entrypoint);

        while let Some(RevOrd((d, idx))) = pq.pop() {
            if res.is_full() && d > res.peek().unwrap().0 {
                break;
            }

            res.push((d, idx));

            let node = &layer[idx];

            for neighbor_idx in node.neighbors.iter().map(|&n| n as usize) {
                if visited.insert(neighbor_idx) {
                    let distance = elements[neighbor_idx].dist(&goal);

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
