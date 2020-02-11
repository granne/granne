# Python Bindings

## Installation

From the repository root run: `pip install .`.

## Basic Usage

Building an index
```
import granne
import numpy as np
np.random.seed(0)

DIMENSION = 100
ELEMENT_TYPE = "angular" # or "angular_int"

builder = granne.GranneBuilder(ELEMENT_TYPE)

for _ in range(10000):
    builder.append(np.random.rand(DIMENSION) - 0.5)

builder.save_elements("elements.bin")

builder.build()

builder.save_index("index.granne")
```

Loading and searching
```
import granne
import numpy as np
np.random.seed(0)

ELEMENT_TYPE = "angular" # or "angular_int"

index = granne.Granne("index.granne", ELEMENT_TYPE, "elements.bin")

DIMENSION = len(index.get_element(0))

for (id, dist) in index.search(np.random.rand(DIMENSION) - 0.5, max_search=150):
    print(f"{id}: {dist}")

```

## Sum Embeddings

With sum embeddings, each element consists of a sum of "embedding" vectors. Each vector is represented by a word (or a label).
Missing words, i.e. words without a corresponding vector, are ignored.

One example could be sentence embeddings created by summing word embeddings. The quality of the sentence embeddings are
highly dependent on the word embeddings. Note that this example uses random vectors in order to showcase the functionality.

```
import granne
import numpy as np
np.random.seed(0)

DIMENSION = 100

embeddings = granne.Embeddings()

for word in "aa bb cc dd ee ff".split():
    embeddings.append(np.random.rand(DIMENSION) - 0.5, word)

embeddings.save_embeddings("embeddings.bin")
embeddings.save_words("words.jl")

builder = granne.GranneBuilder("embeddings", words_path="words.jl", embeddings_path="embeddings.bin", show_progress=False)

for sentence in ["aa bb", "ff aa", "cc dd", "aa aa cc", "bb", "ee ff ff bb"]:
    builder.append(sentence)

builder.build()

builder.save_elements("sentences.bin")
builder.save_index("sentences.granne")

index = granne.Granne("sentences.granne", "embeddings", "sentences.bin", words_path="words.jl", embeddings_path="embeddings.bin")

for (id, dist) in index.search("aa bb cc"):
    sentence = index.get_internal_element(id)
    print(f"{id}: {dist:04f} - {sentence}")

```

## Documentation

```
import granne
help(granne)
```
