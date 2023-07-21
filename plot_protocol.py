from matplotlib import pyplot as plt
import numpy as np
import tikzplotlib

plt.style.use("seaborn-v0_8-white")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
#plt.style.use('ggplot')


x = np.array([0, 1/3., 1/2., 2/3., 1])
chars = ["$0$", r"$\frac{1}{3} \tau$", r"$\frac{1}{2} \tau$", r"$\frac{2}{3} \tau$", r"$\tau$"]

t = np.linspace(0, 1, 100)

reverse = np.piecewise(x, [x <= 1/2, x > 1/2], [lambda x: -x+1, lambda x: x])
pausing = np.piecewise(x, [x <= 1/3,  (x > 1/3) * (x <= 2/3), x > 2/3], [lambda x: -3/2 * x + 1, 1/2, lambda x: x])


fig, ax = plt.subplots()


ax.set_xticks(np.unique(x), chars, fontsize=20)
ax.set_ylim(-0.1, 1.1)
ax.set_yticks(np.arange(-0, 1.1, step=0.1), minor=True)
ax.set_yticks(np.arange(-0, 1.1, step=0.5))
ax.tick_params(axis="y", which="both", labelsize=20)
ax.set_ylabel("$s_t$", fontsize=25)

#plt.title("Annealing Protocols", fontsize=22)
#plt.xlabel(r"annealing time", fontsize=20)

ax.plot(x, reverse, '--', x, pausing, '-.')
ax.legend(labels=["reverse annealing", "reverse + pausing"], fontsize="xx-large", frameon=True, fancybox=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

#fig.show()
plt.savefig("figs/QAv2.pdf")