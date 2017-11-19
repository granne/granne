use std::cmp;
use ordered_float::NotNaN;

pub const DIM: usize = 50;
pub type Element = [f32; DIM];

pub fn dist(x: &Element, y: &Element) -> NotNaN<f32> {
    let mut r = 0.0f32;

    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        r += xi * yi;

        dx += xi * xi;
        dy += yi * yi;
    }

    let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

    return cmp::max(NotNaN::new(0.0f32).unwrap(), d);
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance() {
        let mut a = [0.0f32; DIM];
        let mut b = [0.0f32; DIM];

        a[DIM-1] = 2.0;
        b[DIM-1] = 1.0;

        assert_eq!(NotNaN::new(0.0).unwrap(), dist(&a, &b));
    }

}
