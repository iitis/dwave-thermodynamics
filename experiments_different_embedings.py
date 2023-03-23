import networkx as nx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import dwave.inspector
from minorminer import find_embedding
import copy
import dwave_networkx as dnx
import matplotlib.pyplot as plt

# Setup
qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
target = qpu_sampler.to_networkx_graph()

chain_length = 300
h = {i: 0 for i in range(chain_length)}
J = {(i, i + 1): 1 for i in range(chain_length - 1)}
chain = nx.Graph(J.keys())

middle_label = 1023
def odd(x): return 2 * x + 1


def test_graph(g: nx.Graph):
    plt.figure(figsize=(32, 32))
    dnx.draw_chimera(g, with_labels=True)
    plt.show()


def test_embedding(t: nx.Graph):
    embedding = find_embedding(chain, t)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
    sampleset = sampler.sample_ising(h, J)
    dwave.inspector.show(sampleset)


# First Quadrant
Q1 = copy.deepcopy(target)
for node in target.nodes:
    if node > middle_label or any([True if (node in range(64 * odd(i), 128 * (i + 1))) else False for i in range(8)]):
        Q1.remove_node(node)


# Second Quadrant
Q2 = copy.deepcopy(target)
for node in target.nodes:
    if node > middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8)]):
        Q2.remove_node(node)


# Third Quadrant
Q3 = copy.deepcopy(target)
for node in target.nodes:
    if node <= middle_label or any(
            [True if (node in range(64 * odd(i), 128 * (i+1))) else False for i in range(8, 16)]):
        Q3.remove_node(node)

# Fourth Quadrant
Q4 = copy.deepcopy(target)
for node in target.nodes:
    if node <= middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8, 16)]):
        Q4.remove_node(node)

if __name__ == "__main__":
    for G in [Q4]:
        test_embedding(G)
