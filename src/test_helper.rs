use rand::{self, Rng};

pub fn random_floats() -> impl Iterator<Item = f32> {
    let mut rng = rand::thread_rng();
    std::iter::repeat_with(move || rng.gen::<f32>() - 0.5)
}

pub fn random_vector<Element: std::iter::FromIterator<f32>>(dim: usize) -> Element {
    random_floats().take(dim).collect()
}

pub fn random_vectors<Elements, Element>(dim: usize, num: usize) -> Elements
where
    Elements: std::iter::FromIterator<Element>,
    Element: std::iter::FromIterator<f32>,
{
    (0..num).map(|_| random_vector(dim)).collect()
}

pub fn random_offsets(max_inc: usize) -> impl Iterator<Item = usize> {
    let mut rng = rand::thread_rng();
    let mut cur = 0;
    std::iter::repeat_with(move || {
        cur += rng.gen_range(0, max_inc);
        cur
    })
}

pub fn random_sum_embeddings(
    dim: usize,
    num_embeddings: usize,
    num_elements: usize,
) -> crate::embeddings::SumEmbeddings<'static> {
    let mut sum_embeddings = crate::embeddings::SumEmbeddings::new();
    for _ in 0..num_embeddings {
        sum_embeddings.push_embedding(&random_vector::<Vec<f32>>(dim));
    }

    for i in 0..num_elements {
        let len = 2 + i % 8;
        let element: Vec<_> = (i..(i + len)).map(|j| j % sum_embeddings.num_embeddings()).collect();
        sum_embeddings.push(&element);
    }

    sum_embeddings
}
