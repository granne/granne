#![allow(unused)]

use std::collections::BinaryHeap;

pub struct MaxSizeHeap<T> {
    heap: BinaryHeap<T>,
    max_size: usize,
}

impl<T: Ord> MaxSizeHeap<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_size),
            max_size,
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

    pub fn into_sorted_vec(self: Self) -> Vec<T> {
        self.heap.into_sorted_vec()
    }
}
