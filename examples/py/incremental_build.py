import granne
import numpy as np

DIM = 10
NUM_VECTORS = 10000

vectors = [np.random.rand(DIM) - 0.5 for _ in range(NUM_VECTORS)]

builder = granne.HnswBuilder(num_layers=6, show_progress=True)

for vector in vectors:
    builder.add(vector)

print("Indexing first {} elements into index".format(NUM_VECTORS / 2))
builder.build_index(NUM_VECTORS / 2)

assert NUM_VECTORS == len(builder)
assert NUM_VECTORS/2 == builder.indexed_elements()

builder.save_elements('elements.bin')
builder.save_index('index.granne')

other_builder = granne.HnswBuilder.with_owned_elements(DIM,
                                                       num_layers=6,
                                                       elements_path='elements.bin',
                                                       index_path='index.granne',
                                                       show_progress=True)

assert len(builder) == len(other_builder)
assert builder.indexed_elements() == other_builder.indexed_elements()

print("Continuing...")
print("Indexing the rest of the elements into the index")
other_builder.build_index()

assert len(other_builder) == other_builder.indexed_elements()
