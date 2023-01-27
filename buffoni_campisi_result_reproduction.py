from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding


chain_length = 300
num_samples = 1000

h = {i: 0 for i in range(chain_length - 1)}
