import dwave.inspector
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding
import dwave_networkx as dnx

qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
sampler = EmbeddingComposite(qpu_sampler)


chimera = dnx.chimera_graph(16)
J = {(i, i+1): 1 for i in range(2000)}
embedding = find_embedding(J, qpu_sampler.edgelist)
print(len(embedding))

# h = {i: 1 for i in J.keys()}
#
# sample = sampler.sample_ising(h,J)
#
# dwave.inspector.show(sample)