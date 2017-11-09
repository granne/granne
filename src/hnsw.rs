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
use std::iter;
use std::cmp;
use time;
pub use ordered_float::NotNaN;

const LEVELS: usize = 5;
const LEVEL_MULTIPLIER: usize = 12;

const MAX_NEIGHBORS: usize = 8;
const MAX_INDEX_SEARCH: usize = 200;
const MAX_SEARCH: usize = 200;

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
        layer.extend_from_slice(layers.last().unwrap());

        let start = time::now();

        let begin = layer.len();
        let end = elements.len();

        // insert elements starting at
        for (idx, element) in elements[..end]
            .iter()
            .enumerate()
            .skip(begin)
        {
            let entrypoint = Self::find_entrypoint(&layers,
                                                   &element,
                                                   &elements);

            layer.push(HnswNode {
                neighbors: ArrayVec::new(),
            });

            let neighbors = Self::search_for_neighbors(
                layer,
                &vec![entrypoint],
                elements,
                element,
                MAX_INDEX_SEARCH,
                MAX_NEIGHBORS);

//            println!("Neighbors to {} found: {:?}", idx, neighbors);

            let mut is_connected = false;
            for neighbor in neighbors.into_iter().filter(|&n| n != idx) {
                Self::connect_nodes(layer, elements, idx, neighbor);
                is_connected = is_connected || Self::connect_nodes(layer, elements, neighbor, idx);
            }

            if !is_connected {
                println!("Unconnected node! {}", idx);
//                panic!();
            }

            if idx % 2500 == 0 {
                println!("Added {} vectors in {} s", idx, time::now() - start);
            }
        }
    }

    fn search_for_neighbors(layer: &Vec<HnswNode>,
                            entrypoints: &Vec<usize>,
                            elements: &[Element],
                            goal: &Element,
                            max_search: usize,
                            max_neighbors: usize) -> Vec<usize> {

        let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();
        let mut pq = BinaryHeap::new();
        let mut visited = HashSet::new();

        // push all entrypoints
        for &ep in entrypoints {
            debug_assert!(ep < elements.len());

//            if ep < layer.len()
            {
                pq.push(State {
                    idx: ep,
                    d: NotNaN::new(dist(&elements[ep], &goal)).unwrap()
                });
            }

            visited.insert(ep);
        }

//        println!("New search, ep: {}, max_neighbors: {}", entrypoints[0], max_neighbors);
//        println!("layer length: {}", layer.len());

        let mut num_visited = 0;
        // search for close neighbors
        while let Some(State { idx, d } ) = pq.pop() {
            num_visited += 1;

            if num_visited > max_search {
//                println!("Reached max_search");
                break;
            }

            if res.len() < max_neighbors || d < res.peek().unwrap().0 {
                res.push((d, idx));

                if res.len() > max_neighbors {
                    res.pop();
                }
            }

            let node = &layer[idx];

            for &neighbor_idx in &node.neighbors {
                let neighbor = &elements[neighbor_idx];
                let distance = NotNaN::new(dist(neighbor, &goal)).unwrap();

                if visited.insert(neighbor_idx) {
                    // Remove this check??
                    if res.len() < max_neighbors || d < NotNaN::new(1.5f32).unwrap() * res.peek().unwrap().0
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

    fn find_entrypoint(layers: &[Vec<HnswNode>],
                       element: &Element,
                       elements: &[Element]) -> usize {

        let mut entrypoint = 0;
        for layer in layers {
            let res = Self::search_for_neighbors(
                &layer,
                &vec![entrypoint],
                &elements,
                &element,
                MAX_INDEX_SEARCH,
                1usize);

            entrypoint = res.first().unwrap().clone();
        }

        entrypoint
    }

    fn connect_nodes(layer: &mut Vec<HnswNode>,
                     elements: &[Element],
                     i: usize,
                     j: usize) -> bool
    {
        if layer[i].neighbors.len() < MAX_NEIGHBORS {
            layer[i].neighbors.push(j);
            return true;
        } else {
            let current_distance = NotNaN::new(dist(&elements[i], &elements[j])).unwrap();

            if let Some((k, max_dist)) = layer[i].neighbors
                .iter()
                .map(|&k| NotNaN::new(dist(&elements[i], &elements[k])).unwrap())
                .enumerate()
                .max()
            {
                if current_distance < NotNaN::new(3.0f32).unwrap() * max_dist {
                    layer[i].neighbors[k] = j;
                    return true;
                }
            }
        }

        return false;
    }

    pub fn search(&self, element: &Element) -> Vec<(usize, f32)> {

        let entrypoint = Self::find_entrypoint(&self.levels[..],
                                               element,
                                               &self.elements);

        Self::search_for_neighbors(
            &self.levels[LEVELS-1],
            &vec![entrypoint],
            &self.elements,
            element,
            MAX_SEARCH,
            MAX_NEIGHBORS)
            .iter()
            .map(|&i| (i, dist(&self.elements[i], element))).collect()
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