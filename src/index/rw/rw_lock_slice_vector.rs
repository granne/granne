use crate::slice_vector::FixedWidthSliceVector;
use owning_ref;
use parking_lot::RwLock;

use super::NeighborId;

pub struct RwLockSliceVector {
    data: owning_ref::OwningHandle<Vec<NeighborId>, Vec<RwLock<&'static mut [NeighborId]>>>,
    width: usize,
}

impl RwLockSliceVector {
    pub fn new(data: Vec<NeighborId>, width: usize) -> Self {
        Self {
            data: owning_ref::OwningHandle::new_with_fn(data, |d| unsafe {
                (*(d as *mut [NeighborId])).chunks_mut(width).map(RwLock::new).collect()
            }),
            width,
        }
    }

    pub fn as_slice(self: &Self) -> &[RwLock<&'static mut [NeighborId]>] {
        &*self.data
    }

    pub fn len(self: &Self) -> usize {
        (*self.data).len()
    }

    // this is unsafe because the underlying data may be modified through as_slice
    pub unsafe fn as_owner(self: &Self) -> &[NeighborId] {
        self.data.as_owner()
    }

    pub fn into_owner(self: Self) -> Vec<NeighborId> {
        self.data.into_owner()
    }
}

impl Into<FixedWidthSliceVector<'static, NeighborId>> for RwLockSliceVector {
    fn into(self: Self) -> FixedWidthSliceVector<'static, NeighborId> {
        let width = self.width;
        FixedWidthSliceVector::with_data(self.into_owner(), width)
    }
}

impl From<FixedWidthSliceVector<'static, NeighborId>> for RwLockSliceVector {
    fn from(fw_vec: FixedWidthSliceVector<'static, NeighborId>) -> Self {
        assert!(fw_vec.width() > 0);

        let width = fw_vec.width();
        let data: Vec<NeighborId> = fw_vec.into();

        RwLockSliceVector::new(data, width)
    }
}
