//
// concurrent, little waiting (X)
// mmap (X)
// build layer by layer (X)
// extenstible (X)
// small size (X)
// fast
// merge indexes?
//

use arrayvec::ArrayVec;
use fnv::FnvHashSet;
use ordered_float::NotNaN;
use revord::RevOrd;
use std::collections::BinaryHeap;
use std::cmp;
use types::*;
use pbr::ProgressBar;

// Write and read
use std::fs::File;
use std::io::{Write, Result};

// Threading
use rayon::prelude::*;
use std::sync::{Mutex, RwLock};

const MAX_NEIGHBORS: usize = 32;
type NeighborType = u32;

#[repr(C)]
#[derive(Clone, Default, Debug)]
struct HnswNode {
    neighbors: ArrayVec<[NeighborType; MAX_NEIGHBORS]>,
}


pub struct Config {
    pub num_layers: usize,
    pub layer_multiplier: usize,
    pub max_search: usize,
    pub show_progress: bool,
}


pub struct HnswBuilder<'a, T: HasDistance + Sync + Send + 'a> {
    layers: Vec<Vec<HnswNode>>,
    elements: &'a [T],
    config: Config,
}


pub struct Hnsw<'a, T: HasDistance + 'a> {
    layers: Vec<&'a [HnswNode]>,
    elements: &'a [T],
}


impl<'a, T: HasDistance + Sync + Send + 'a> HnswBuilder<'a, T> {

    pub fn new(config: Config, elements: &'a [T]) -> Self {
        assert!(elements.len() < <NeighborType>::max_value() as usize);

        HnswBuilder {
            layers: Vec::new(),
            elements: elements,
            config: config,
        }
    }


    pub fn save_to_disk(self: &Self, path: &str) {

        let mut file = File::create(path).unwrap();

        self.write(&mut file);
    }


    pub fn write<B: Write>(self: &Self, buffer: &mut B) -> Result<()> {
        let num_nodes = self.layers.iter().map(|layer| layer.len()).sum();
        let num_layers = self.layers.len();
        let layer_counts = self.layers.iter().map(|layer| layer.len());

        let mut usize_data = vec![num_nodes, num_layers];
        usize_data.extend(layer_counts);

        let data = unsafe {
            ::std::slice::from_raw_parts(
                usize_data.as_ptr() as *const u8,
                usize_data.len() * ::std::mem::size_of::<usize>())
        };

        buffer.write(data)?;

        for layer in &self.layers {

            let data = unsafe {
                ::std::slice::from_raw_parts(
                    layer.as_ptr() as *const u8,
                    layer.len() * ::std::mem::size_of::<HnswNode>())
            };

            buffer.write(data)?;
        }

        Ok(())
    }


    pub fn get_index(self: &'a Self) -> Hnsw<'a, T> {
        Hnsw {
            layers: self.layers
                .iter()
                .map(|layer| &layer[..])
                .collect(),
            elements: self.elements,
        }
    }


    pub fn load(config: Config, index: &Hnsw<T>, elements: &'a [T]) -> Self {
        let mut builder = Self::new(config, elements);

        assert!(index.layers.last().unwrap().len() <= elements.len());

        builder.layers = index.layers.iter()
            .map(|layer| layer.to_vec())
            .collect();

        builder
    }


    pub fn build_index(&mut self) {
        self.layers.push(vec![HnswNode::default()]);

        let mut num_elements_in_layer = 1;
        for layer in 1..self.config.num_layers {
            num_elements_in_layer *= self.config.layer_multiplier;

            if num_elements_in_layer > self.elements.len() ||
               layer == self.config.num_layers - 1
            {
                num_elements_in_layer = self.elements.len();
            }

            // copy layer above
            let mut new_layer = Vec::with_capacity(num_elements_in_layer);
            new_layer.extend_from_slice(self.layers.last().unwrap());

            Self::insert_elements(&self.config,
                                  &self.elements[..num_elements_in_layer],
                                  &self.get_index(),
                                  &mut new_layer);

            self.layers.push(new_layer);

            if num_elements_in_layer == self.elements.len() {
                break;
            }
        }
    }


    pub fn append_elements(&mut self, elements: &'a [T]) {
        assert!(elements.len() < <NeighborType>::max_value() as usize);

        assert!(self.elements[0].dist(&elements[0]).into_inner() <
                DIST_EPSILON);

        assert!(self.elements[self.elements.len()-1].dist(
                     &elements[self.elements.len()-1]).into_inner() <
                DIST_EPSILON);

        self.elements = elements;

        let mut layer = self.layers.pop().unwrap();

        Self::insert_elements(&self.config,
                              self.elements,
                              &self.get_index(),
                              &mut layer);

        self.layers.push(layer);
    }


    fn insert_elements(config: &Config,
                       elements: &[T],
                       prev_layers: &Hnsw<T>,
                       layer: &mut Vec<HnswNode>) {

        assert!(layer.len() <= elements.len());

        let already_inserted = layer.len();

        layer.resize(elements.len(), HnswNode::default());

        // create RwLocks for underlying nodes
        let layer: Vec<RwLock<&mut HnswNode>> =
            layer.iter_mut()
            .map(|node| RwLock::new(node))
            .collect();

        // set up progress bar
        let step_size = cmp::max(100, elements.len() / 400);
        let progress_bar = {
            if config.show_progress {
                let mut progress_bar = ProgressBar::new(elements.len() as u64);
                let info_text = format!("Layer {}: ", prev_layers.layers.len());
                progress_bar.message(&info_text);
                progress_bar.set((step_size * (already_inserted / step_size)) as u64);

                Some(Mutex::new(progress_bar))

            } else {
                None
            }
        };

        // insert elements, skipping already inserted
        elements.par_iter()
            .enumerate()
            .skip(already_inserted)
            .for_each(
                |(idx, _)| {
                    Self::insert_element(config,
                                         elements,
                                         prev_layers,
                                         &layer,
                                         idx);

                    // This only shows approximate progress because of par_iter
                    if idx % step_size == 0 {
                        if let Some(ref progress_bar) = progress_bar {
                            progress_bar.lock().unwrap().add(step_size as u64);
                        }
                    }
                }
            );

        if let Some(progress_bar) = progress_bar {
            progress_bar.lock().unwrap().finish_println("");
        }
    }


    fn insert_element(config: &Config,
                      elements: &[T],
                      prev_layers: &Hnsw<T>,
                      layer: &Vec<RwLock<&mut HnswNode>>,
                      idx: usize) {

        let element = &elements[idx];
        let (entrypoint, _) = prev_layers.search(element, config.max_search / 10)[0];

        let neighbors = Self::search_for_neighbors_index(elements,
                                                         &layer[..],
                                                         entrypoint,
                                                         element,
                                                         config.max_search);

        let neighbors =
            Self::select_neighbors(elements, neighbors, MAX_NEIGHBORS);

        Self::initialize_node(&layer[idx], &neighbors[..]);

        for (neighbor, d) in neighbors {
            Self::connect_nodes(elements, &layer[neighbor], neighbor, idx, d);
        }
    }


    // Similar to Hnsw::search_for_neighbors but with RwLocks for
    // parallel insertion
    fn search_for_neighbors_index(elements: &[T],
                                  layer: &[RwLock<&mut HnswNode>],
                                  entrypoint: usize,
                                  goal: &T,
                                  max_search: usize)
                                  -> Vec<(usize, NotNaN<f32>)> {

        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> =
            MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited = FnvHashSet::default();

        pq.push(RevOrd(
            (elements[entrypoint].dist(&goal), entrypoint)
        ));

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


    fn select_neighbors(elements: &[T],
                        candidates: Vec<(usize, NotNaN<f32>)>,
                        max_neighbors: usize) -> Vec<(usize, NotNaN<f32>)> {

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
            if neighbors.iter().all(|&(k, _)| d < elements[j].dist(&elements[k])) {
                neighbors.push((j, d));
            } else {
                pruned.push((j, d));
            }
        }

        let remaining = max_neighbors - neighbors.len();
        neighbors.extend(pruned.into_iter().take(remaining));

        neighbors
    }


    fn initialize_node(node: &RwLock<&mut HnswNode>,
                       neighbors: &[(usize, NotNaN<f32>)]) {
        // Write Lock!
        let mut node = node.write().unwrap();

        debug_assert!(node.neighbors.len() == 0);
        let num_to_add =
            node.neighbors.capacity() - node.neighbors.len();

        for &(idx, _) in neighbors.iter().take(num_to_add) {
            node.neighbors.push(idx as NeighborType);
        }
    }


    fn connect_nodes(elements: &[T],
                     node: &RwLock<&mut HnswNode>,
                     i: usize,
                     j: usize,
                     d: NotNaN<f32>)
    {
        // Write Lock!
        let mut node = node.write().unwrap();

        if node.neighbors.len() < MAX_NEIGHBORS {
            node.neighbors.push(j as NeighborType);
        } else {

            let mut candidates: Vec<_> = node.neighbors.iter()
                .map(|&k| (k as usize, elements[i].dist(&elements[k as usize])))
                .collect();

            candidates.push((j as usize, d));
            candidates.sort_unstable_by_key(|&(_, d)| d);

            let neighbors =
                Self::select_neighbors(elements, candidates, MAX_NEIGHBORS);

            for (k, (n, _)) in neighbors.into_iter().enumerate() {
                node.neighbors[k] = n as NeighborType;
            }
        }
    }
}


impl<'a, T: HasDistance + 'a> Hnsw<'a, T> {

    pub fn load(buffer: &'a [u8], elements: &'a [T]) -> Self {

        let offset = 0 * ::std::mem::size_of::<usize>();
        let num_nodes = &buffer[offset] as *const u8 as *const usize;

        let offset = 1 * ::std::mem::size_of::<usize>();
        let num_layers = &buffer[offset] as *const u8 as *const usize;

        let offset = 2 * ::std::mem::size_of::<usize>();

        let layer_counts: &[usize] = unsafe {
            ::std::slice::from_raw_parts(
                &buffer[offset] as *const u8 as *const usize,
                *num_layers
        )};

        let offset = (2 + layer_counts.len()) * ::std::mem::size_of::<usize>();

        let nodes: &[HnswNode] = unsafe {
            ::std::slice::from_raw_parts(
                &buffer[offset] as *const u8 as *const HnswNode,
                *num_nodes
            )
        };

        let mut layers = Vec::new();

        let mut start = 0;
        for &layer_count in layer_counts {
            let end = start + layer_count;
            let layer = &nodes[start..end];
            layers.push(layer);
            start = end;
        }

        assert!(layers.last().unwrap().len() <= elements.len());

        Self {
            layers: layers,
            elements: elements,
        }
    }


    pub fn search(&self, element: &T, max_search: usize) -> Vec<(usize, f32)> {

        let (bottom_layer, top_layers) = self.layers.split_last().unwrap();

        let entrypoint = Self::find_entrypoint(&top_layers,
                                               element,
                                               &self.elements,
                                               cmp::max(10, max_search / 5));

        let num_neighbors = 5;

        Self::search_for_neighbors(
            &bottom_layer,
            entrypoint,
            &self.elements,
            element,
            max_search)
            .into_iter()
            .take(num_neighbors)
            .map(|(i, d)| (i, d.into_inner())).collect()
    }


    fn find_entrypoint(layers: &[&[HnswNode]],
                       element: &T,
                       elements: &[T],
                       max_search: usize) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res = Self::search_for_neighbors(
                &layer,
                entrypoint,
                &elements,
                &element,
                max_search);

            entrypoint = res[0].0;
        }

        entrypoint
    }


    fn search_for_neighbors(layer: &[HnswNode],
                            entrypoint: usize,
                            elements: &[T],
                            goal: &T,
                            max_search: usize)
                            -> Vec<(usize, NotNaN<f32>)> {


        let mut res: MaxSizeHeap<(NotNaN<f32>, usize)> =
            MaxSizeHeap::new(max_search);
        let mut pq: BinaryHeap<RevOrd<_>> = BinaryHeap::new();
        let mut visited = FnvHashSet::default();

        pq.push(RevOrd(
            (elements[entrypoint].dist(&goal), entrypoint)
        ));

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


mod tests {
    use super::*;
    use std::mem;
    use types::example::*;

    #[test]
    fn test_hnsw_node_size()
    {
        assert!((MAX_NEIGHBORS) * mem::size_of::<NeighborType>() <=
                mem::size_of::<HnswNode>());

        assert!(mem::size_of::<HnswNode>() <=
                MAX_NEIGHBORS * mem::size_of::<NeighborType>()
                + mem::size_of::<usize>());
    }

    #[test]
    fn test_select_neighbors()
    {
        let element = random_float_element();

        let other_elements: Vec<FloatElement> =
            (0..50).map(|_| random_float_element()).collect();

        let candidates: Vec<_> = other_elements
            .iter()
            .map(|e| e.dist(&element))
            .enumerate()
            .collect();

        let neighbors =
            HnswBuilder::select_neighbors(&other_elements[..],
                                          candidates.clone(),
                                          10);

        assert_eq!(10, neighbors.len());


        let neighbors = HnswBuilder::select_neighbors(&other_elements[..],
                                                      candidates.clone(),
                                                      60);

        assert_eq!(50, neighbors.len());
    }

    #[test]
    fn more_layers_than_needed()
    {
        let elements: Vec<FloatElement> =
            (0..100).map(|_| random_float_element()).collect();

        let config = Config {
            num_layers: 6,
            layer_multiplier: 4,
            max_search: 10,
            show_progress: false,
        };

        let mut builder = HnswBuilder::new(config, &elements[..]);
        builder.build_index();

        assert_eq!(5, builder.layers.len());
        assert_eq!(elements.len(), builder.layers[4].len());
    }

    #[test]
    fn fewer_layers_than_needed()
    {
        let elements: Vec<FloatElement> =
            (0..100).map(|_| random_float_element()).collect();

        let config = Config {
            num_layers: 4,
            layer_multiplier: 4,
            max_search: 10,
            show_progress: false,
        };

        let mut builder = HnswBuilder::new(config, &elements[..]);
        builder.build_index();

        assert_eq!(4, builder.layers.len());
        assert_eq!(elements.len(), builder.layers[3].len());
    }

    #[test]
    fn write_and_load()
    {
        let elements: Vec<FloatElement> =
            (0..100).map(|_| random_float_element()).collect();

        let config = Config {
            num_layers: 4,
            layer_multiplier: 6,
            max_search: 10,
            show_progress: false,
        };

        let mut builder = HnswBuilder::new(config, &elements[..]);
        builder.build_index();

        let mut data = Vec::new();
        builder.write(&mut data);

        let index = Hnsw::load(&data[..], &elements[..]);

        assert_eq!(builder.layers.len(), index.layers.len());

        for layer in 0..builder.layers.len() {
            assert_eq!(builder.layers[layer].len(), index.layers[layer].len());

            for i in 0..builder.layers[layer].len() {
                assert_eq!(builder.layers[layer][i].neighbors,
                           index.layers[layer][i].neighbors);
            }
        }
    }

    #[test]
    fn append_elements() {
        let elements: Vec<FloatElement> =
            (0..200).map(|_| random_float_element()).collect();

        let config = Config {
            num_layers: 4,
            layer_multiplier: 6,
            max_search: 10,
            show_progress: false,
        };

        // insert half of the elements
        let mut builder = HnswBuilder::new(config, &elements[..100]);
        builder.build_index();

        assert_eq!(4, builder.layers.len());
        assert_eq!(100, builder.layers[3].len());

        let max_search = 10;

        // assert that one arbitrary element is findable (might fail)
        {
            let index = builder.get_index();

            assert!(index.search(&elements[50], max_search)
                    .iter()
                    .any(|&(idx, _)| 50 == idx));
        }

        // insert rest of the elements
        builder.append_elements(&elements[..]);

        assert_eq!(4, builder.layers.len());
        assert_eq!(200, builder.layers[3].len());

        // assert that the same arbitrary element and a newly added one
        // is findable (might fail)
        {
            let index = builder.get_index();

            assert!(index.search(&elements[50], max_search)
                    .iter()
                    .any(|&(idx, _)| 50 == idx));

            assert!(index.search(&elements[150], max_search)
                    .iter()
                    .any(|&(idx, _)| 150 == idx));
        }
    }
}
