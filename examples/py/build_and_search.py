import granne
import numpy as np

DIM = 128
NUM_VECTORS = 10000
NUM_LAYERS = 5

vectors = [np.random.rand(DIM) - 0.5 for _ in range(NUM_VECTORS)]
test_vector = np.random.rand(DIM) - 0.5
builder = granne.HnswBuilder(DIM, NUM_LAYERS)

for vector in vectors:
    builder.add(vector)

print("Building index")
builder.build_index()
print("Index built\n")

# use directly
print("Searching for test vector:")
results = builder.search(test_vector)

for i, d in results:
    print("Vector #{} with distance d={}".format(i, d))
print

# or save index and vectors to disk and use later
INDEX_PATH = "test_index.granne"
ELEMENTS_PATH = "test_vectors.bin"

print("Saving to disk")
builder.save_index(INDEX_PATH)
builder.save_elements(ELEMENTS_PATH)

print("Loading index")
index = granne.Hnsw(INDEX_PATH, ELEMENTS_PATH, DIM)

print("Searching for test vector:")
results = index.search(test_vector)

for i, d in results:
    print("Vector #{} with distance d={}".format(i, d))
