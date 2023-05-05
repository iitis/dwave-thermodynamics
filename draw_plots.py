import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

cwd = os.getcwd()

beta = 1
#dataset = f"results_eff_Q1_reverse"
#dataset = f"results_eff_Q1_pausing"
#dataset = f"results_eff_reverse_beta_1_s_0.5"
dataset = f"results_eff_pausing_beta_1_s_0.5"
#dataset = f"results_eff_reverse_with_h_beta_1_s_0.41_h_1"


if __name__ == "__main__":
    with open(os.path.join(cwd, "results", f"{dataset}.pkl"), "rb") as f:
        prob, dist = pickle.load(f)

    fig, ax1 = plt.subplots()
    #title = f"Reverse annealing without pausing Q1 \n for beta = {beta}"
    #plt.title(title)

    color1 = "tab:blue"
    ax1.set_xlabel(r"$t (\mu s)$", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.set_ylabel(r"$\mathcal{P}_{GS}$", fontsize=20)
    ax1.plot(list(prob.keys()), list(prob.values()), color=color1)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=20)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = "tab:red"
    ax2.set_ylabel(r"$\mathcal{F}_{GS}$", fontsize=20)
    ax2.plot(list(dist.keys()), [v for v in dist.values()], color=color2)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig("figs/PF_pausing.pdf")
    #plt.savefig("figs/PF_Q1_pausing.pdf")
    #plt.savefig(os.path.join(cwd, "results", "plots", f"{dataset}.png"))
