
pub trait Array<T> {
    fn as_slice(self: &Self) -> &[T];
    fn as_mut_slice(self: &mut Self) -> &mut [T];
}

macro_rules! impl_array(
    ($t:ty, $len:expr) => (
        impl Array<$t> for [$t; $len] {
            #[inline(always)]
            fn as_slice(self: &Self) -> &[$t] {
                &self[..]
            }

            #[inline(always)]
            fn as_mut_slice(self: &mut Self) -> &mut [$t] {
                &mut self[..]
            }
        }
    );
);

macro_rules! impl_arrays {
    ($t:ty, $len:expr) => (impl_array!($t, $len););
    ($t:ty, $len:expr, $($tail:expr),+) => (
        impl_array!($t, $len);
        impl_arrays!($t, $($tail),+);
    )
}

impl_arrays!(f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30, 32, 50, 60, 64, 96, 100, 128, 200, 256, 300);
impl_arrays!(i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30, 32, 50, 60, 64, 96, 100, 128, 200, 256, 300);
