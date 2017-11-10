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
        layer.resize(elements.len(), HnswNode::default());

        let start = time::now();

        let already_inserted = layers.last().unwrap().len();

        // insert elements starting at begin
        for (idx, element) in elements
            .iter()
            .enumerate()
            .skip(already_inserted)
        {
            let entrypoint = Self::find_entrypoint(&layers,
                                                   &element,
                                                   &elements);

            let neighbors = Self::search_for_neighbors(layer,
                                                       entrypoint,
                                                       elements,
                                                       element,
                                                       MAX_INDEX_SEARCH,
                                                       MAX_NEIGHBORS);

            for neighbor in neighbors.into_iter().filter(|&n| n != idx) {
                // can be done directly since layer[idx].neighbors is empty
                Self::connect_nodes(&mut layer[idx], elements, idx, neighbor);

                // find a more clever way to decide when to add this edge
                Self::connect_nodes(&mut layer[neighbor], elements, neighbor, idx);
            }

            if idx % 2500 == 0 {
                println!("Added {} vectors in {} s", idx, time::now() - start);
            }
        }
    }

    fn search_for_neighbors(layer: &Vec<HnswNode>,
                            entrypoint: usize,
                            elements: &[Element],
                            goal: &Element,
                            max_search: usize,
                            max_neighbors: usize) -> Vec<usize> {

        let mut res: BinaryHeap<(NotNaN<f32>, usize)> = BinaryHeap::new();
        let mut pq = BinaryHeap::new();
        let mut visited = HashSet::new();

        pq.push(State {
            idx: entrypoint,
            d: dist(&elements[entrypoint], &goal)
        });

        visited.insert(entrypoint);

        let mut num_visited = 0;
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
                let distance = dist(neighbor, &goal);

                if visited.insert(neighbor_idx) {
                    // Remove this check??
                    if res.len() < max_neighbors || d <= NotNaN::new(3f32).unwrap() * res.peek().unwrap().0
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
                entrypoint,
                &elements,
                &element,
                MAX_INDEX_SEARCH,
                1usize);

            entrypoint = res.first().unwrap().clone();
        }

        entrypoint
    }


    fn connect_nodes(node: &mut HnswNode,
                     elements: &[Element],
                     i: usize,
                     j: usize) -> bool
    {
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
