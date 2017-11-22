//
// concurrent, little waiting (X)
// mmap (X)
// build layer by layer (X)
// small size
// extenstible (X)
// merge indexes?
// fast
//

use types::*;
use arrayvec::ArrayVec;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::iter;
use std::cmp;
use time;
use std::mem;
pub use ordered_float::NotNaN;

// Write and read
use std::fs::File;
use std::io::prelude::*;
use memmap::Mmap;
use revord::RevOrd;

// Threading
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

use fnv::FnvHashSet;

const MAX_NEIGHBORS: usize = 20;

#[repr(C)]
#[derive(Clone, Default, Debug)]
struct HnswNode {
    neighbors: ArrayVec<[usize; MAX_NEIGHBORS]>,
}


pub struct Config {
    pub num_levels: usize,
    pub level_multiplier: usize,
    pub max_search: usize,
}


pub struct HnswBuilder<'a, T: HasDistance + Sync + 'a> {
    levels: Vec<Vec<HnswNode>>,
    elements: &'a [T],
    config: Config,
}


pub struct Hnsw<'a, T: HasDistance + 'a> {
    levels: Vec<&'a [HnswNode]>,
    elements: &'a [T],
}


impl<'a, T: 'a + HasDistance + Sync> HnswBuilder<'a, T> {

    pub fn new(config: Config, elements: &'a [T]) -> Self {
        HnswBuilder {
            levels: Vec::new(),
            elements: elements,
            config: config,
        }
    }


    pub fn save_to_disk(self: &Self, path: &str) {

        let mut file = File::create(path).unwrap();

        self.write(&mut file);
    }


    pub fn write<B: Write>(self: &Self, buffer: &mut B) {
        let num_nodes = self.levels.iter().map(|level| level.len()).sum();
        let num_levels = self.levels.len();
        let level_counts = self.levels.iter().map(|level| level.len());

        let mut usize_data = vec![num_nodes, num_levels];
        usize_data.extend(level_counts);

        let data = unsafe {
            ::std::slice::from_raw_parts(
                usize_data.as_ptr() as *const u8,
                usize_data.len() * ::std::mem::size_of::<usize>())
        };

        buffer.write(data);

        for level in &self.levels {

            let data = unsafe {
                ::std::slice::from_raw_parts(
                    level.as_ptr() as *const u8,
                    level.len() * ::std::mem::size_of::<HnswNode>())
            };

            buffer.write(data);
        }
    }


    pub fn load(config: Config, index: &Hnsw<T>, elements: &'a [T]) -> Self {
        let mut builder = Self::new(config, elements);

        assert!(index.levels.last().unwrap().len() <= elements.len());

        builder.levels = index.levels.iter()
            .map(|level| level.to_vec())
            .collect();

        builder
    }


    pub fn build_index(&mut self) {
        self.levels.push(vec![HnswNode::default()]);

        let mut num_elements = 1;
        for level in 1..self.config.num_levels {
            num_elements *= self.config.level_multiplier;
            num_elements = cmp::min(num_elements, self.elements.len());

            // copy layer above
            let mut new_layer = Vec::with_capacity(num_elements);
            new_layer.extend_from_slice(self.levels.last().unwrap());

            Self::insert_elements(&self.config,
                                  &mut new_layer,
                                  &self.levels[..],
                                  &self.elements[..num_elements]);

            self.levels.push(new_layer);
        }
    }

    pub fn append_elements(&mut self, elements: &'a [T]) {
        assert!(self.elements[0].dist(&elements[0]) <
                NotNaN::new(::std::f32::EPSILON).unwrap());

        assert!(self.elements[self.elements.len()-1].dist(
                     &elements[self.elements.len()-1]) <
                NotNaN::new(::std::f32::EPSILON).unwrap());

        self.elements = elements;

        let (layer, layers) = self.levels.split_last_mut().unwrap();
        Self::insert_elements(&self.config, layer, layers, self.elements);
    }


    fn insert_elements(config: &Config,
                       layer: &mut Vec<HnswNode>,
                       layers: &[Vec<HnswNode>],
                       elements: &[T]) {

        println!("Building layer {} with {} vectors", layers.len(), elements.len());

        assert!(layer.len() <= elements.len());

        let already_inserted = layer.len();

        layer.resize(elements.len(), HnswNode::default());

        // create RwLocks for underlying nodes
        let layer: Vec<RwLock<&mut HnswNode>> =
            layer.iter_mut()
            .map(|node| RwLock::new(node))
            .collect();

        // insert elements, skipping already inserted
        layer
            .par_iter()
            .enumerate()
            .skip(already_inserted)
            .for_each(
                |(idx, _)| {
                    Self::insert_element(config,
                                         layers,
                                         &layer,
                                         elements,
                                         idx);
                });
    }


    fn insert_element(config: &Config,
                      layers: &[Vec<HnswNode>],
                      layer: &Vec<RwLock<&mut HnswNode>>,
                      elements: &[T],
                      idx: usize) {

        let element = &elements[idx];
        let entrypoint = Self::find_entrypoint(layers,
                                               element,
                                               elements,
                                               config.max_search);

        let neighbors = Self::search_for_neighbors_index(&layer[..],
                                                         entrypoint,
                                                         elements,
                                                         element,
                                                         config.max_search,
                                                         MAX_NEIGHBORS);

        for (neighbor, d) in neighbors.into_iter().filter(|&(n, _)| n != idx) {
            // can be done directly since layer[idx].neighbors is empty
            Self::connect_nodes(&layer[idx], elements, idx, neighbor, d);

            // find a more clever way to decide when to add this edge
            Self::connect_nodes(&layer[neighbor], elements, neighbor, idx, d);
        }
    }


    fn search_for_neighbors_index(layer: &[RwLock<&mut HnswNode>],
                                  entrypoint: usize,
                                  elements: &[T],
                                  goal: &T,
                                  max_search: usize,
                                  max_neighbors: usize) -> Vec<(usize, NotNaN<f32>)> {

        let mut res = MaxSizeHeap::new(max_neighbors);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited = FnvHashSet::default();

        pq.push(RevOrd(
            (elements[entrypoint].dist(&goal), entrypoint)
        ));

        visited.insert(entrypoint);

        for _ in 0..max_search {

            if let Some(RevOrd((d, idx))) = pq.pop() {
                res.push((d, idx));

                let node = layer[idx].read().unwrap();

                for &neighbor_idx in &node.neighbors {
                    if visited.insert(neighbor_idx) {
                        let distance = elements[neighbor_idx].dist(&goal);
                        pq.push(RevOrd((distance, neighbor_idx)));
                    }
                }

            } else {
                break;
            }
        }

        return res.heap.into_vec().into_iter().map(|(d, idx)| (idx, d)).collect();
    }


    fn connect_nodes(node: &RwLock<&mut HnswNode>,
                     elements: &[T],
                     i: usize,
                     j: usize,
                     d: NotNaN<f32>) -> bool
    {
        // Write Lock!
        let mut node = node.write().unwrap();

        if node.neighbors.len() < MAX_NEIGHBORS {
            node.neighbors.push(j);
            return true;
        } else {
            if let Some((k, max_dist)) = node.neighbors
                .iter()
                .map(|&k| elements[i].dist(&elements[k]))
                .enumerate()
                .max()
            {
                if d < NotNaN::new(2.0f32).unwrap() * max_dist {
                    node.neighbors[k] = j;
                    return true;
                }
            }
        }

        return false;
    }


    fn find_entrypoint(layers: &[Vec<HnswNode>],
                       element: &T,
                       elements: &[T],
                       max_search: usize) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res = search_for_neighbors(
                &layer,
                entrypoint,
                &elements,
                &element,
                max_search,
                1usize);

            entrypoint = res.first().unwrap().0.clone();
        }

        entrypoint
    }
}


impl<'a, T: HasDistance + 'a> Hnsw<'a, T> {

    pub fn load(buffer: &'a [u8], elements: &'a [T]) -> Self {

        let offset = 0 * ::std::mem::size_of::<usize>();
        let num_nodes = &buffer[offset] as *const u8 as *const usize;

        let offset = 1 * ::std::mem::size_of::<usize>();
        let num_levels = &buffer[offset] as *const u8 as *const usize;

        let offset = 2 * ::std::mem::size_of::<usize>();

        let level_counts: &[usize] = unsafe {
            ::std::slice::from_raw_parts(
                &buffer[offset] as *const u8 as *const usize,
                *num_levels
        )};

        let offset = (2 + level_counts.len()) * ::std::mem::size_of::<usize>();

        let nodes: &[HnswNode] = unsafe {
            ::std::slice::from_raw_parts(
                &buffer[offset] as *const u8 as *const HnswNode,
                *num_nodes
            )
        };

        let mut levels = Vec::new();

        let mut start = 0;
        for &level_count in level_counts {
            let end = start + level_count;
            let level = &nodes[start..end];
            levels.push(level);
            start = end;
        }

        assert!(levels.last().unwrap().len() <= elements.len());

        Self {
            levels: levels,
            elements: elements,
        }

    }


    pub fn search(&self, element: &T, max_search: usize) -> Vec<(usize, f32)> {

        let (bottom_level, top_levels) = self.levels.split_last().unwrap();

        let entrypoint = Self::find_entrypoint(&top_levels,
                                               element,
                                               &self.elements,
                                               max_search);

        search_for_neighbors(
            &bottom_level,
            entrypoint,
            &self.elements,
            element,
            max_search,
            MAX_NEIGHBORS)
            .into_iter()
            .map(|(i, d)| (i, d.into_inner())).collect()
    }


    fn find_entrypoint(layers: &[&[HnswNode]],
                       element: &T,
                       elements: &[T],
                       max_search: usize) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res = search_for_neighbors(
                &layer,
                entrypoint,
                &elements,
                &element,
                max_search,
                1usize);

            entrypoint = res.first().unwrap().0.clone();
        }

        entrypoint
    }
}


fn search_for_neighbors<T: HasDistance>(layer: &[HnswNode],
                                        entrypoint: usize,
                                        elements: &[T],
                                        goal: &T,
                                        max_search: usize,
                                        max_neighbors: usize)
                                        -> Vec<(usize, NotNaN<f32>)> {

    let mut res = MaxSizeHeap::new(max_neighbors);
    let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
    let mut visited = FnvHashSet::default();

    pq.push(RevOrd(
        (elements[entrypoint].dist(&goal), entrypoint)
    ));

    visited.insert(entrypoint);

    for _ in 0..max_search {

        if let Some(RevOrd((d, idx))) = pq.pop() {
            res.push((d, idx));

            let node = &layer[idx];

            for &neighbor_idx in &node.neighbors {
                if visited.insert(neighbor_idx) {
                    let distance = elements[neighbor_idx].dist(&goal);
                    pq.push(RevOrd((distance, neighbor_idx)));
                }
            }

        } else {
            break;
        }
    }

    return res.heap.into_sorted_vec().into_iter().map(|(d, idx)| (idx, d)).collect();
}


struct MaxSizeHeap<T> {
    heap: BinaryHeap<T>,
    max_size: usize
}

impl<T: Ord> MaxSizeHeap<T> {

    pub fn new(max_size: usize) -> Self {
        MaxSizeHeap {
            heap: BinaryHeap::with_capacity(max_size),
            max_size: max_size
        }
    }

    pub fn push(self: &mut Self, element: T) {
        if self.heap.len() < self.max_size {
            self.heap.push(element);
        }
        else if element < *self.heap.peek().unwrap() {
            if self.heap.len() >= self.max_size {
                self.heap.pop();
            }

            self.heap.push(element);
        }
    }

    pub fn peek(self: &Self) -> Option<&T> {
        self.heap.peek()
    }

    pub fn len(self: &Self) -> usize {
        self.heap.len()
    }
}


mod tests {
    use super::*;

    #[test]
    fn test_hnsw_node_size()
    {
        assert!((MAX_NEIGHBORS) * mem::size_of::<usize>() < mem::size_of::<HnswNode>());
    }

    #[test]
    fn test_hnsw()
    {

    }
}
