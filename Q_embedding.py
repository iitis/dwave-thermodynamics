from dwave.system import DWaveSampler
import dwave.inspector
import networkx as nx
from minorminer import find_embedding
from dwave.system.composites import FixedEmbeddingComposite
from copy import deepcopy

def odd(x): return 2 * x + 1

chain_length = 300
h = {i: 0 for i in range(chain_length)}
J = {(i, i + 1): 1 for i in range(chain_length - 1)}

qpu_sampler = DWaveSampler(solver='DW_2000Q_6',token="DEV-0ca1a227b2e2e90bffff07bb12ef35c2a00d28c6")
target = qpu_sampler.to_networkx_graph()
middle_label = 1023

Q1 = deepcopy(target)
for node in target.nodes:
    if node > middle_label or any([True if (node in range(64 * odd(i), 128 * (i + 1))) else False for i in range(8)]):
        Q1.remove_node(node)
        
Q2 = deepcopy(target)
for node in target.nodes:
    if node > middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8)]):
        Q2.remove_node(node)

    # Third Quadrant
Q3 = deepcopy(target)
for node in target.nodes:
    if node <= middle_label or any(
            [True if (node in range(64 * odd(i), 128 * (i+1))) else False for i in range(8, 16)]):
        Q3.remove_node(node)

    # Fourth Quadrant
Q4 = deepcopy(target)
for node in target.nodes:
    if node <= middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8, 16)]):
        Q4.remove_node(node)


chain = nx.Graph(J.keys())
embedding = find_embedding(chain, Q1)
sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
response = sampler.sample_ising(h, J, num_reads=100)

dwave.inspector.show(response)