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

pub fn random_sum_embeddings() -> crate::elements::SumEmbeddings<'static> {
    let embeddings: crate::elements::angular::Vectors = (0..225)
        .map(|_| random_vector::<crate::elements::angular::Vector>(25))
        .collect();

    let mut sum_embeddings = crate::elements::SumEmbeddings::new(embeddings);

    for i in 0..101 {
        let len = 2 + i % 8;
        let element: Vec<_> = (i..(i + len))
            .map(|j| j % sum_embeddings.num_embeddings())
            .collect();
        sum_embeddings.push(&element);
    }

    sum_embeddings
}
