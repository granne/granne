use std::cmp;

pub const DIM: usize = 50;
pub type Scalar = f32;
pub type Element = [Scalar; DIM];

pub fn dist(x: &Element, y: &Element) -> Scalar {
    let mut r = Scalar::default();

    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        r += xi * yi;

        dx += xi * xi;
        dy += yi * yi;
    }

    return 1.0f32 - (r / (dx.sqrt() * dy.sqrt()));

//    return cmp::min(0.0f32, 1.0f32 - (r / (dx.sqrt() * dy.sqrt())));
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance() {
        let mut a = [Scalar::default(); DIM];
        let mut b = [Scalar::default(); DIM];

        a[DIM-1] = 2.0;
        b[DIM-1] = 1.0;

        assert_eq!(2.0, dist(&a, &b));
    }
}
