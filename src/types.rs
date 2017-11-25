use std::cmp;
use ordered_float::NotNaN;
use rblas;

pub const DIM: usize = 100;
pub const DIST_EPSILON: f32 = 10.0 * ::std::f32::EPSILON;

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

        let r: f32 = rblas::Dot::dot(&x[..], &y[..]);
        let dx: f32 = rblas::Nrm2::nrm2(&x[..]);
        let dy: f32 = rblas::Nrm2::nrm2(&y[..]);

        let d = NotNaN::new(1.0f32 - (r / (dx * dy))).unwrap();

        cmp::max(NotNaN::new(0.0f32).unwrap(), d)
    }
}

pub fn reference_dist(first: &FloatElement, second: &FloatElement) -> NotNaN<f32>
{
    let &FloatElement(x) = first;
    let &FloatElement(y) = second;

    let r: f32 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi as f32 * yi as f32).sum();
    let dx: f32 = x.iter().map(|&xi| xi as f32 * xi as f32).sum();
    let dy: f32 = y.iter().map(|&yi| yi as f32 * yi as f32).sum();

    let d = NotNaN::new(1.0f32 - (r / (dx.sqrt() * dy.sqrt()))).unwrap();

    cmp::max(NotNaN::new(0.0f32).unwrap(), d)
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
    fn test_rblas_dist() {
        for _ in 0..100 {
            let x = example::random_float_element();
            let y = example::random_float_element();

            assert!((x.dist(&y) - reference_dist(&x, &y)).abs() < DIST_EPSILON);
        }
    }

    #[test]
    fn test_dist_between_same_vector() {
        for _ in 0..100 {
            let x = example::random_float_element();

            assert!(x.dist(&x).into_inner() < DIST_EPSILON);
        }
    }
}
