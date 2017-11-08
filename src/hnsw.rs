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
use std::mem;
use time;
pub use ordered_float::NotNaN;

const LEVELS: usize = 1;

const MAX_NEIGHBORS: usize = 50;
const MAX_INDEX_SEARCH: usize = 500;
const MAX_SEARCH: usize = 5000;

#[derive(Default)]
struct HnswNode {
    element: usize,
    neighbors: ArrayVec<[usize; MAX_NEIGHBORS]>,
}


pub struct Hnsw {
    levels: [Vec<HnswNode>; LEVELS],
    entrypoints: Vec<usize>,
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


impl Hnsw {
    pub fn new(elements: Vec<Element>) -> Self {
        Hnsw {
            levels: [Vec::with_capacity(elements.len())],
            entrypoints: (0..2500).step_by(100).collect(),
            elements: elements,
        }
    }

    pub fn build_index(&mut self) {
        

        for level in 0..LEVELS {
            Self::build_layer(&mut self.levels[level], 
                              &self.entrypoints, 
                              &self.elements);
        }
    }

    pub fn search(&self, element: &Element) -> Vec<(usize, f32)> {
        let LAYER = 0;
        Self::search_for_neighbors(
            &self.levels[LAYER],
            &self.entrypoints,
            &self.elements,
            element,
            MAX_SEARCH,
            MAX_NEIGHBORS)
            .iter()
            .map(|&i| (i, dist(&self.elements[i], element))).collect()
    }

    fn build_layer(layer: &mut Vec<HnswNode>, 
                   entrypoints: &Vec<usize>, 
                   elements: &Vec<Element>) {

        println!("Building layer with {} vectors", elements.len());

        let start = time::now();
        for (idx, element) in elements.iter().enumerate() {

            layer.push(HnswNode {
                element: idx,
                neighbors: ArrayVec::new(),
            });

            let neighbors = Self::search_for_neighbors(
                layer, 
                entrypoints, 
                elements, 
                element, 
                MAX_INDEX_SEARCH,
                MAX_NEIGHBORS);

            for neighbor in neighbors.into_iter().filter(|&n| n != idx) {
                Self::connect_nodes(layer, elements, idx, neighbor);
                Self::connect_nodes(layer, elements, neighbor, idx);
            }

            if idx % 2500 == 0 {
                println!("Added {} vectors in {} s", idx, time::now() - start);
            }
        }
    }

    fn search_for_neighbors(layer: &Vec<HnswNode>,
                            entrypoints: &Vec<usize>,
                            elements: &Vec<Element>,
                            goal: &Element,
                            max_search: usize,
                            max_neighbors: usize) -> Vec<usize> {

        let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();
        let mut pq = BinaryHeap::new();
        let mut visited = HashSet::new();

        // push all entrypoints
        for &ep in entrypoints {
            debug_assert!(ep < elements.len());

            if ep < layer.len() {
                pq.push(State { 
                    idx: ep,
                    d: NotNaN::new(dist(&elements[ep], &goal)).unwrap()
                });
            }

            visited.insert(ep);
        }

        let mut num_visited = 0;
        // search for close neighbors
        while let Some(State { idx, d } ) = pq.pop() {
            num_visited += 1;

            if num_visited > max_search {
                println!("Reached max_search");
                break;
            }

            if res.len() < max_neighbors || d < res.peek().unwrap().0 {
                res.push((d, idx));
            }

            if res.len() > max_neighbors {
                res.pop();
            }

            let node = &layer[idx];
 
            for &neighbor_idx in &node.neighbors {
                let neighbor = &elements[neighbor_idx];
                let distance = NotNaN::new(dist(neighbor, &goal)).unwrap();

                if visited.insert(neighbor_idx) {
                    // Remove this check??
                    if res.len() < max_neighbors || d < res.peek().unwrap().0 
                    {
                        pq.push(State {
                            idx: neighbor_idx,
                            d: distance,
                        });
                    }
                }
            }
        }

        return res.into_sorted_vec().into_iter().map(|(_, idx)| idx).collect();
    }

    fn connect_nodes(layer: &mut Vec<HnswNode>,
                     elements: &Vec<Element>,
                     i: usize,
                     j: usize)
    {
//        let distance = NotNaN::new(dist(&elements[i], &elements[j])).unwrap();
//        println!("Connecting {} and {} with distance {}", i, j, distance);

        if layer[i].neighbors.len() < MAX_NEIGHBORS {
            layer[i].neighbors.push(j);
        } else {
            let current_distance = NotNaN::new(dist(&elements[i], &elements[j])).unwrap();

            if let Some((k, min_dist)) = layer[i].neighbors
                .iter()
                .map(|&k| NotNaN::new(dist(&elements[i], &elements[k])).unwrap())
                .enumerate()
                .min()
            {
                if current_distance < min_dist {
                    layer[i].neighbors[k] = j;
                }
            }
        }
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
