import matplotlib.pyplot as plt
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler

c16 = dnx.chimera_graph(16)

qpu_sampler = DWaveSampler(solver='DW_2000Q_6') #DW_2000Q_2_1 or DW_2000Q_5(lower noise)
target = qpu_sampler.to_networkx_graph()


plt.figure(figsize=(64, 64))
dnx.draw_chimera(target, with_labels=True)
plt.savefig("Chimera")
plt.show()

