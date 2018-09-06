use bytes::{ByteOrder, LittleEndian};

const BYTES_PER_NEIGHBOR_ID: usize = 5;

#[repr(C,packed)]
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct NeighborId(pub [u8; BYTES_PER_NEIGHBOR_ID]);

impl From<usize> for NeighborId {
    fn from(integer: usize) -> Self {
        let mut data: [u8; BYTES_PER_NEIGHBOR_ID] = unsafe { ::std::mem::uninitialized() };
        LittleEndian::write_uint(&mut data, integer as u64, BYTES_PER_NEIGHBOR_ID);
        NeighborId(data)
    }
}


impl Into<usize> for NeighborId {
    #[inline(always)]
    fn into(self: Self) -> usize {
        LittleEndian::read_uint(&self.0, BYTES_PER_NEIGHBOR_ID) as usize
    }
}

impl NeighborId {
    pub fn max_value() -> usize {
        let data = [<u8>::max_value(); BYTES_PER_NEIGHBOR_ID];
        NeighborId(data).into()
    }
}

#[cfg(test)]
mod tests {
    use std::mem;
    use super::*;

    #[test]
    fn hnsw_node_size() {
        assert_eq!(BYTES_PER_NEIGHBOR_ID, mem::size_of::<NeighborId>());
    }
}
