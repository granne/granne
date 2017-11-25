use std::cmp;
use ordered_float::NotNaN;

pub const DIM: usize = 100;

pub trait HasDistance {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32>;
}

#[repr(C)]
pub struct FloatElement([f32; DIM]);

impl From<[f32; DIM]> for FloatElement {
    fn from(array: [f32; DIM]) -> FloatElement {
        FloatElement (array)
    }
}

impl HasDistance for FloatElement {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32>
    {
        let &FloatElement(x) = self;
        let &FloatElement(y) = other;

        let r: f32 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi as f32 * yi as f32).sum();
        let dx: f32 = x.iter().map(|&xi| xi as f32 * xi as f32).sum();
        let dy: f32 = y.iter().map(|&yi| yi as f32 * yi as f32).sum();

        let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}

#[repr(C)]
pub struct Int8Element([u8; DIM]);

impl HasDistance for Int8Element {
    fn dist(self: &Self, other: &Self) -> NotNaN<f32>
    {
        let &Int8Element(x) = self;
        let &Int8Element(y) = other;

        let r: i32 = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi as i32 * yi as i32)
            .sum();

        let dx: i32 = x.iter()
            .map(|&xi| xi as i32 * xi as i32)
            .sum();

        let dy: i32 = y.iter()
            .map(|&yi| yi as i32 * yi as i32)
            .sum();

        let dx = dx as f32;
        let dy = dy as f32;

        let d = NotNaN::new(1.0f32 - (r as f32 / (dx.sqrt() * dy.sqrt()))).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}


pub mod example {
    use super::*;
    use rand;
    use rand::Rng;

    pub fn random_float_element() -> FloatElement {
        let mut rng = rand::thread_rng();

        let mut data = [0.0f32; DIM];

        for f in &mut data[..] {
            *f = rng.gen();
        }

        data.into()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance() {
    }
}
