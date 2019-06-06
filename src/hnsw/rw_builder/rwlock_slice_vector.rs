use parking_lot::RwLock;
use slice_vector::FixedWidthSliceVector;

pub struct RwLockSliceVector<T: 'static + Clone> {
    width: usize,
    data: Vec<RwLock<&'static mut [T]>>,
    data_ptr: *mut T,
}

unsafe impl<T: Clone + Send + Sync> Send for RwLockSliceVector<T> {}
unsafe impl<T: Clone + Send + Sync> Sync for RwLockSliceVector<T> {}

impl<T: Clone> From<FixedWidthSliceVector<'static, T>> for RwLockSliceVector<T> {
    fn from(fw_vec: FixedWidthSliceVector<'static, T>) -> Self {
        let width = fw_vec.width();
        let mut data: Vec<_> = fw_vec.into();
        let data_ptr = data.as_mut_ptr();

        // into_boxed_slice will drop any excess capacity
        let rw_data = Box::leak(data.into_boxed_slice())
            .chunks_mut(width)
            .map(|elem| RwLock::new(elem))
            .collect();

        Self {
            width: width,
            data: rw_data,
            data_ptr,
        }
    }
}

impl<T: Clone> Into<FixedWidthSliceVector<'static, T>> for RwLockSliceVector<T> {
    fn into(self: Self) -> FixedWidthSliceVector<'static, T> {
        FixedWidthSliceVector::from_vec(self.width, self.drain().into())
    }
}

impl<T: Clone> RwLockSliceVector<T> {
    pub fn as_slice<'a>(self: &'a Self) -> &'a [RwLock<&'static mut [T]>] {
        self.data.as_slice()
    }

    pub fn len(self: &Self) -> usize {
        self.data.len()
    }

    fn drain(mut self: Self) -> Vec<T> {
        let data_size = self.width * self.data.len();
        self.data.clear();

        unsafe { Vec::<T>::from_raw_parts(self.data_ptr, data_size, data_size) }
    }
}

impl<T: Clone> Drop for RwLockSliceVector<T> {
    fn drop(self: &mut Self) {
        let data_size = self.width * self.data.len();
        self.data.clear();

        unsafe {
            Vec::<T>::from_raw_parts(self.data_ptr, data_size, data_size);
        }
    }
}
