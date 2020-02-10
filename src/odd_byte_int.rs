use byteorder::{ByteOrder, LittleEndian};

macro_rules! oddbyte_int {
    ($type_name:ident, $num_bytes:expr) => {
        /// An integer type representing an offset/id using $num_bytes bytes for
        /// improved compression. Stores integer no larger than 256**$num_bytes
        #[repr(C, packed)]
        #[derive(Clone, Copy, Eq, PartialEq)]
        pub struct $type_name([u8; $num_bytes]);

        impl $type_name {
            pub const fn max_value() -> Self {
                Self([0xFF; $num_bytes])
            }
        }

        impl Into<usize> for $type_name {
            #[inline(always)]
            fn into(self: Self) -> usize {
                LittleEndian::read_uint(&self.0, $num_bytes) as usize
            }
        }

        impl From<usize> for $type_name {
            #[inline(always)]
            fn from(integer: usize) -> Self {
                let mut data = [0u8; $num_bytes];
                LittleEndian::write_uint(&mut data, integer as u64, $num_bytes);
                $type_name(data)
            }
        }
    };
}

oddbyte_int!(ThreeByteInt, 3);
oddbyte_int!(FiveByteInt, 5);

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;

    #[test]
    fn into_into() {
        let original: usize = 123456;
        let neighborid: ThreeByteInt = original.into();
        let converted: usize = neighborid.into();

        assert_eq!(original, converted);
    }

    #[test]
    fn query_offset_conversions() {
        for &integer in &[
            0usize,
            1usize,
            2usize,
            3usize,
            1234567usize,
            usize::try_from(FiveByteInt::max_value()).unwrap() - 1,
            usize::try_from(FiveByteInt::max_value()).unwrap(),
        ] {
            let query_offset: FiveByteInt = integer.into();
            let into_integer: usize = query_offset.into();
            assert_eq!(integer, into_integer);
        }
    }

    #[test]
    fn query_offset_cast() {
        let integer: usize = 7_301_010_345;
        let query_offset: FiveByteInt = integer.into();

        let reinterpreted = &query_offset.0[0] as *const u8 as *const FiveByteInt;
        let reinterpreted = unsafe { *reinterpreted };

        assert_eq!(integer, usize::try_from(reinterpreted).unwrap());
    }
}
