use crate::WordDict;
use cpython::{FromPyObject, PyObject, PyResult};
use granne::{self, Dist, ElementContainer};
use std::cell::RefCell;

type Element = <granne::embeddings::SumEmbeddings<'static> as ElementContainer>::Element;

py_class!(pub class Embeddings |py| {
    data embeddings: RefCell<granne::embeddings::SumEmbeddings<'static>>;
    data words: RefCell<WordDict>;

    // Required since rust-cpython cannot add docs for "special" functions
    /// Note: This is the documentation for the `__new__` method:
    /// Constructs a new Embeddings object.
    ///
    /// Parameters
    /// ----------
    /// Optional:
    /// embeddings_path: str
    ///     Path to embeddings
    /// words_path: str
    ///     Path to words
    ///
    @classmethod
    def __init__(_cls) -> PyResult<PyObject> { Ok(py.None()) }

    def __new__(_cls,
                embeddings_path: Option<String> = None,
                words_path: Option<String> = None
    ) -> PyResult<Embeddings> {

        let embeddings = embeddings_path
            .map(|path| std::fs::File::open(path).expect("Could not open embeddings file"));

        let (embeddings, words) = match (embeddings, words_path) {
            (Some(_), None) => {panic!("embeddings_path specifiec, but not words_path");},
            (None, Some(_)) => {panic!("words_path specified, but not embeddings_path!");},
            (Some(embeddings), Some(words)) => {
                let words = WordDict::new(&words);
                let embeddings = unsafe {
                    granne::embeddings::SumEmbeddings::from_files(&embeddings, None)
                        .expect("Could not load embeddings.")
                };

                (embeddings, words)
            },
            (None, None) => {
                let embeddings = granne::embeddings::SumEmbeddings::new();
                let words = WordDict::default();

                (embeddings, words)
            },
        };

        Embeddings::create_instance(py, RefCell::new(embeddings), RefCell::new(words))
    }

    def __len__(&self) -> PyResult<usize> {
        Ok(self.embeddings(py).borrow().num_embeddings())
    }

    /// Returns the (non-normalized) embedding for a word/sentence or id/list of ids.
    def get_embedding(&self, input: &PyObject) -> PyResult<Vec<f32>>
    {
        let idxs: Vec<usize> = if let Ok(idx) = usize::extract(py, input) {
            vec![idx]
        } else if let Ok(data) = Vec::<usize>::extract(py, input) {
            data
        } else {
            let words = String::extract(py, input)?;
            self.words(py).borrow().get_word_ids(&words)
        };

        Ok(self.embeddings(py).borrow().create_embedding(&idxs))
    }

    /// Computes the (angular) distance between left_input and right_input.
    def dist(&self, left_input: &PyObject, right_input: &PyObject) -> PyResult<f32> {
        let left = Element::from(self.get_embedding(py, left_input).unwrap());
        let right = Element::from(self.get_embedding(py, right_input).unwrap());

        Ok(left.dist(&right).into_inner())
    }

    /// Computes the (angular) distance between the left_input and each element of right_inputs.
    def dists(&self, left_input: &PyObject, right_inputs: Vec<PyObject>) -> PyResult<Vec<f32>> {
        let left = Element::from(self.get_embedding(py, left_input).unwrap());
        let rights: Vec<Element> = right_inputs
            .into_iter()
            .map(|i| Element::from(self.get_embedding(py, &i).unwrap()))
            .collect();

        Ok(rights.iter().map(|r| left.dist(r).into_inner()).collect())
    }

    /// Appends an embedding toghether with a word to this collection.
    /// Returns true if the insertion is successful.
    /// Returns false if the word already exists, in which case the embedding will not be inserted.
    ///
    /// Parameters
    /// ----------
    /// Required:
    /// embedding: [f32]
    ///     An embedding vector
    /// word: str
    ///     The word or label representing the vector
    ///
    def append(&self, embedding: Vec<f32>, word: String) -> PyResult<bool> {
        let inserted = self.words(py).borrow_mut().push(word);
        if inserted {
            self.embeddings(py).borrow_mut().push_embedding(&embedding);
        }

        Ok(inserted)
    }

    /// Saves the embeddings (vectors) and words/labels to disk.
    def save(&self, embeddings_path: &str, words_path: &str) -> PyResult<PyObject> {
        self.save_embeddings(py, embeddings_path)?;
        self.save_words(py, words_path)?;

        Ok(py.None())
    }

    /// Saves the embeddings (vectors) to disk.
    def save_embeddings(&self, path: &str) -> PyResult<PyObject> {
        let mut file = std::fs::File::create(path)
            .expect("Could not create file!");
        self.embeddings(py).borrow().write_embeddings(&mut file)
            .expect("Could not write embeddings!");

        Ok(py.None())
    }

    /// Saves the words/labels to disk.
    def save_words(&self, path: &str) -> PyResult<PyObject> {
        let mut file = std::fs::File::create(path).expect("Could not create file!");

        // todo what format?
        self.words(py).borrow().write(&mut file).expect("Could not write words!");

        Ok(py.None())
    }
});
