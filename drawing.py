import matplotlib.pyplot as plt
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from copy import deepcopy

c16 = dnx.chimera_graph(16)

qpu_sampler = DWaveSampler(solver='DW_2000Q_6') #DW_2000Q_2_1 or DW_2000Q_5(lower noise)
target = qpu_sampler.to_networkx_graph()

Q1 = deepcopy(target)
for node in target.nodes:
    if node > middle_label or any([True if (node in range(64 * odd(i), 128 * (i + 1))) else False for i in range(8)]):
        Q1.remove_node(node)

plt.figure(figsize=(64, 64))
dnx.draw_chimera(Q1, with_labels=True)
#plt.savefig("Chimera")
plt.show()

