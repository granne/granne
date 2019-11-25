use rand::{self, Rng};

pub fn random_floats() -> impl Iterator<Item = f32> {
    let mut rng = rand::thread_rng();
    std::iter::repeat_with(move || rng.gen::<f32>() - 0.5)
}

pub fn random_vector<Element: std::iter::FromIterator<f32>>(dim: usize) -> Element {
    random_floats().take(dim).collect()
}

pub fn random_offsets(max_inc: usize) -> impl Iterator<Item = usize> {
    let mut rng = rand::thread_rng();
    let mut cur = 0;
    std::iter::repeat_with(move || {
        cur += rng.gen_range(0, max_inc);
        cur
    })
}
