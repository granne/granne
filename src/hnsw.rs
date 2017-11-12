//
// concurrent, little waiting
// mmap
// small size
// extenstible (possibly not indefinately)
// merge indexes?
// build layer by layer
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

use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use fnv::FnvHashSet;

const LEVELS: usize = 5;
const LEVEL_MULTIPLIER: usize = 12;

const MAX_NEIGHBORS: usize = 15;
const MAX_INDEX_SEARCH: usize = 500;
const MAX_SEARCH: usize = 500;

#[derive(Clone, Default, Debug)]
struct HnswNode {
    neighbors: ArrayVec<[usize; MAX_NEIGHBORS]>,
}


pub struct Hnsw {
    levels: [Vec<HnswNode>; LEVELS],
    elements: Vec<Element>,
}


#[derive(Copy, Clone, PartialEq, Eq)]
struct State {
    d: NotNaN<f32>,
    idx: usize,
}


impl Ord for State {
    fn cmp(&self, other: &State) -> Ordering {
        other.d.cmp(&self.d)
    }
}


impl PartialOrd for State {
    fn partial_cmp(&self, other: &State) -> Option<Ordering> {
        other.d.partial_cmp(&self.d)
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




impl Hnsw {
    pub fn new(elements: Vec<Element>) -> Self {
        Hnsw {
//            levels: [Vec::with_capacity(elements.len()); LEVELS],
            levels: iter::repeat(Vec::new())
                .collect::<ArrayVec<_>>()
                .into_inner()
                .unwrap(),
            elements: elements,
        }
    }

    pub fn build_index(&mut self) {
        self.levels[0].push(HnswNode {
            neighbors: ArrayVec::new(),
        });

        let mut num_elements = 1;
        for level in 1..LEVELS {
            num_elements *= LEVEL_MULTIPLIER;
            num_elements = cmp::min(num_elements, self.elements.len());

            let (new_layer, top_layers) =
                self.levels[..level+1].split_last_mut().unwrap();

            Self::build_layer(top_layers,
                              new_layer,
                              &self.elements[..num_elements]);
        }
    }

    fn build_layer(layers: &[Vec<HnswNode>],
                   layer: &mut Vec<HnswNode>,
                   elements: &[Element]) {

        println!("Building layer {} with {} vectors", layers.len(), elements.len());

        // copy layer above
        *layer = Vec::with_capacity(elements.len());
        layer.extend_from_slice(layers.last().unwrap());
        layer.resize(elements.len(), HnswNode::default());

        let already_inserted = layers.last().unwrap().len();

        // create RwLocks for underlying nodes
        let layer: Vec<_> =
            layer.iter_mut()
                .map(|node| RwLock::new(node))
                .collect();

        // insert elements skipping already inserted
        elements
            .par_iter()
            .enumerate()
            .skip(already_inserted)
            .for_each(
                |(idx, element)| {
                    Self::insert_element(layers,
                                         &layer,
                                         elements,
                                         idx);
                });
    }

    fn insert_element(layers: &[Vec<HnswNode>],
                      layer: &Vec<RwLock<&mut HnswNode>>,
                      elements: &[Element],
                      idx: usize) {

        let element = &elements[idx];
        let entrypoint = Self::find_entrypoint(layers,
                                               element,
                                               elements);

        let neighbors = Self::search_for_neighbors_index(&layer,
                                                         entrypoint,
                                                         elements,
                                                         element,
                                                         MAX_INDEX_SEARCH,
                                                         MAX_NEIGHBORS);

        for neighbor in neighbors.into_iter().filter(|&n| n != idx) {
            // can be done directly since layer[idx].neighbors is empty
            Self::connect_nodes(&layer[idx], elements, idx, neighbor);

            // find a more clever way to decide when to add this edge
            Self::connect_nodes(&layer[neighbor], elements, neighbor, idx);
        }
    }



    fn search_for_neighbors_index(layer: &Vec<RwLock<&mut HnswNode>>,
                                 entrypoint: usize,
                                 elements: &[Element],
                                 goal: &Element,
                                 max_search: usize,
                                 max_neighbors: usize) -> Vec<usize> {

        let mut res = MaxSizeHeap::new(max_neighbors);
        let mut pq = BinaryHeap::new();
        let mut visited = FnvHashSet::default();

        pq.push(State {
            idx: entrypoint,
            d: dist(&elements[entrypoint], &goal)
        });

        visited.insert(entrypoint);

        for _ in 0..max_search {

            if let Some(State { idx, d } ) = pq.pop() {

                res.push((d, idx));

                // Read Lock!
                let node = layer[idx].read().unwrap();

                for &neighbor_idx in &node.neighbors {
                    if visited.insert(neighbor_idx) {
                        pq.push(State {
                            idx: neighbor_idx,
                            d: dist(&elements[neighbor_idx], &goal),
                        });
                    }
                }

            } else {
                break;
            }
        }

        return res.heap.into_vec().into_iter().map(|(_, idx)| idx).collect();
    }


    fn connect_nodes(node: &RwLock<&mut HnswNode>,
                     elements: &[Element],
                     i: usize,
                     j: usize) -> bool
    {
        // Write Lock!
        let mut node = node.write().unwrap();

        if node.neighbors.len() < MAX_NEIGHBORS {
            node.neighbors.push(j);
            return true;
        } else {
            let current_distance =
                dist(&elements[i], &elements[j]);

            if let Some((k, max_dist)) = node.neighbors
                .iter()
                .map(|&k| dist(&elements[i], &elements[k]))
                .enumerate()
                .max()
            {
                if current_distance < NotNaN::new(2.0f32).unwrap() * max_dist {
                    node.neighbors[k] = j;
                    return true;
                }
            }
        }

        return false;
    }


    fn search_for_neighbors(layer: &Vec<HnswNode>,
                            entrypoint: usize,
                            elements: &[Element],
                            goal: &Element,
                            max_search: usize,
                            max_neighbors: usize) -> Vec<usize> {

        let mut res = MaxSizeHeap::new(max_neighbors);
        let mut pq = BinaryHeap::new();
        let mut visited = HashSet::new();

        pq.push(State {
            idx: entrypoint,
            d: dist(&elements[entrypoint], &goal)
        });

        visited.insert(entrypoint);

        for _ in 0..max_search {

            if let Some(State { idx, d } ) = pq.pop() {

                res.push((d, idx));

                let node = &layer[idx];

                for &neighbor_idx in &node.neighbors {
                    if visited.insert(neighbor_idx) {
                        pq.push(State {
                            idx: neighbor_idx,
                            d: dist(&elements[neighbor_idx], &goal),
                        });
                    }
                }

            } else {
                break;
            }
        }

        return res.heap.into_sorted_vec().into_iter().map(|(_, idx)| idx).collect();
    }


    fn find_entrypoint(layers: &[Vec<HnswNode>],
                       element: &Element,
                       elements: &[Element]) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res = Self::search_for_neighbors(
                &layer,
                entrypoint,
                &elements,
                &element,
                MAX_INDEX_SEARCH,
                1usize);

            entrypoint = res.first().unwrap().clone();
        }

        entrypoint
    }


    pub fn search(&self, element: &Element) -> Vec<(usize, f32)> {

        let entrypoint = Self::find_entrypoint(&self.levels[..],
                                               element,
                                               &self.elements);

        Self::search_for_neighbors(
            &self.levels[LEVELS-1],
            entrypoint,
            &self.elements,
            element,
            MAX_SEARCH,
            MAX_NEIGHBORS)
            .iter()
            .map(|&i| (i, dist(&self.elements[i], element).into_inner())).collect()
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
