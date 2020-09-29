#[cfg(feature = "blas")]
use blas;

#[cfg(not(feature = "blas"))]
pub fn dot_product_f32(x: &[f32], y: &[f32]) -> f32 {
    // optimized code to compute the dot product for systems supporting avx2
    // with fallback for other systems

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_product_avx2(x: &[f32], y: &[f32]) -> f32 {
        // this function will be inlined and take advantage of avx2 auto-vectorization
        dot_product_fallback(x, y)
    }

    #[inline(always)]
    fn dot_product_fallback(x: &[f32], y: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 32;
        let mut chunk = [0.0f32; CHUNK_SIZE];

        for (a, b) in x.chunks_exact(CHUNK_SIZE).zip(y.chunks_exact(CHUNK_SIZE)) {
            for i in 0..CHUNK_SIZE {
                chunk[i] = a[i].mul_add(b[i], chunk[i]);
            }
        }

        let mut r = 0.0f32;
        for i in 0..CHUNK_SIZE {
            r += chunk[i];
        }

        for (ai, bi) in x
            .chunks_exact(CHUNK_SIZE)
            .remainder()
            .iter()
            .zip(y.chunks_exact(CHUNK_SIZE).remainder())
        {
            r = ai.mul_add(*bi, r);
        }

        r
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(x, y) };
        }
    }

    dot_product_fallback(x, y)
}

#[cfg(feature = "blas")]
pub fn dot_product_f32(x: &[f32], y: &[f32]) -> f32 {
    unsafe { blas::sdot(x.len() as i32, x, 1, y, 1) }
}

pub fn dot_product_and_squared_norms_i8(x: &[i8], y: &[i8]) -> (i32, i32, i32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_r_dx_dy_avx2(x: &[i8], y: &[i8]) -> (i32, i32, i32) {
        compute_r_dx_dy_fallback(x, y)
    }

    #[inline(always)]
    fn compute_r_dx_dy_fallback(x: &[i8], y: &[i8]) -> (i32, i32, i32) {
        let mut r = 0i32;
        let mut dx = 0i32;
        let mut dy = 0i32;

        for (xi, yi) in x.iter().map(|&xi| i32::from(xi)).zip(y.iter().map(|&yi| i32::from(yi))) {
            r += xi * yi;
            dx += xi * xi;
            dy += yi * yi;
        }

        (r, dx, dy)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_r_dx_dy_avx2(x, y) };
        }
    }

    compute_r_dx_dy_fallback(x, y)
}

#[cfg(not(feature = "blas"))]
pub fn sum_into_f32(x: &mut [f32], y: &[f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_into_avx2(x: &mut [f32], y: &[f32]) {
        sum_into_fallback(x, y)
    }

    #[inline(always)]
    fn sum_into_fallback(x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len());

        for i in 0..x.len() {
            x[i] += y[i];
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { sum_into_avx2(x, y) };
        }
    }

    sum_into_fallback(x, y)
}

#[cfg(feature = "blas")]
pub fn sum_into_f32(x: &mut [f32], y: &[f32]) {
    unsafe { blas::saxpy(x.len() as i32, 1f32, y, 1, x, 1) };
}

#[cfg(not(feature = "blas"))]
pub fn normalize_f32(x: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn normalize_avx2(x: &mut [f32]) {
        normalize_fallback(x)
    }

    #[inline(always)]
    fn normalize_fallback(x: &mut [f32]) {
        let norm = dot_product_f32(x, x).sqrt();

        if norm > 0.0 {
            for i in 0..x.len() {
                x[i] /= norm;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { normalize_avx2(x) };
        }
    }

    normalize_fallback(x)
}

#[cfg(feature = "blas")]
pub fn normalize_f32(x: &mut [f32]) {
    let n = x.len() as i32;
    let norm = unsafe { blas::snrm2(n, x, 1) };
    if norm > 0.0 {
        unsafe { blas::sscal(n, 1.0 / norm, x, 1) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helper;

    #[test]
    fn sum() {
        for i in 1..101 {
            let mut x: Vec<f32> = test_helper::random_floats().take(i).collect();
            let y: Vec<f32> = test_helper::random_floats().take(i).collect();

            let mut expected = x.clone();
            for (i, xi) in expected.iter_mut().enumerate() {
                *xi += y[i];
            }

            sum_into_f32(&mut x, &y);

            assert_eq!(expected, x);
        }
    }

    #[test]
    fn dot_product() {
        for i in 1..101 {
            let x: Vec<f32> = test_helper::random_floats().take(i).collect();
            let y: Vec<f32> = test_helper::random_floats().take(i).collect();

            let mut expected = 0.0f32;
            for i in 0..i {
                expected += x[i] * y[i];
            }

            assert!((expected - dot_product_f32(&x, &y)).abs() < 0.000001f32);
        }
    }
}
