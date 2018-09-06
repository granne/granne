use fnv::FnvHasher;
use std::hash::Hasher;
use bit_vec::BitVec;

pub struct BloomFilter {
    table: BitVec,
    mask: usize,
}

impl BloomFilter {
    pub fn new(num_elements: usize, accepted_error: f64) -> Self {
        let num_bits = (num_elements as f64 / accepted_error).log2().ceil() as usize;
        let size = 1 << num_bits;

        Self {
            table: BitVec::from_elem(size, false),
            mask: size - 1,
        }
    }


    pub fn insert(self: &mut Self, x: usize) -> bool {
        let h = Self::hash(x) & self.mask;

        let already_inserted = self.table.get(h).unwrap();
        self.table.set(h, true);

        !already_inserted
    }


    #[allow(dead_code)]
    pub fn contains(self: &Self, x: usize) -> bool {
        let h = Self::hash(x) & self.mask;
        self.table.get(h).unwrap()
    }


    #[inline(always)]
    fn hash(x: usize) -> usize {
        let mut hasher = FnvHasher::default();

        hasher.write(&Self::get_bytes(x));

        hasher.finish() as usize
    }


    #[inline(always)]
    fn get_bytes(x: usize) -> [u8; 8] {
        [
            x as u8,
            (x >> 8) as u8,
            (x >> 16) as u8,
            (x >> 24) as u8,
            (x >> 32) as u8,
            (x >> 40) as u8,
            (x >> 48) as u8,
            (x >> 56) as u8,
        ]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_same() {
        let mut h = BloomFilter::new(100, 0.01);

        assert!(h.insert(123));
        assert!(!h.insert(123));

        assert!(h.contains(123));
    }

    #[test]
    fn insert_multiple() {
        let mut h = BloomFilter::new(100, 0.01);

        assert!(h.insert(123));
        assert!(h.insert(321));
        assert!(!h.insert(123));
        assert!(!h.insert(321));

        assert!(h.contains(123));
        assert!(h.contains(321));
    }

    #[test]
    fn size() {
        for &num_elements in &[100, 1000, 2, 8, 1024, 1337, 961234] {
            for &error in &[0.05, 0.5, 0.001, 0.005] {
                let h = BloomFilter::new(num_elements, error);
                assert!(num_elements as f64 <= h.table.len() as f64 * error,
                        format!("{} vs {}", num_elements as f64, h.table.len() as f64 * error));
                assert!(h.table.len() as f64 * error <= 2.0 * num_elements as f64,
                        format!("{} vs {}", h.table.len() as f64 * error, 2.0 * num_elements as f64));
            }
        }
    }

    #[test]
    fn error_rate() {
        for &num_elements in &[100, 1000, 2, 8, 1024, 1337, 961234] {
            for &error in &[0.05, 0.5, 0.001, 0.005] {
                let mut h = BloomFilter::new(num_elements, error);

                for i in 0..num_elements {
                    h.insert(i);
                }

                let mut hits = 0;
                let to_check: Vec<_> = (num_elements..(num_elements + 1000)).collect();
                for &i in &to_check {
                    if h.contains(i) {
                        hits += 1;
                    }
                }

                assert!(hits as f64 / to_check.len() as f64 <= 2.0 * error,
                        format!("num_elements: {}, error: {}, hits: {}, tested: {}", num_elements, error, hits, to_check.len()));
            }
        }
    }

}
