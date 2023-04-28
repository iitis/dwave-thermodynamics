import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

cwd = os.getcwd()

beta = 1
dataset = f"results_eff_reverse_with_h_beta_1_s_0.41_h_1"

if __name__ == "__main__":
    with open(os.path.join(cwd, "results", f"{dataset}.pkl"), "rb") as f:
        prob, dist = pickle.load(f)

    fig, ax1 = plt.subplots()
    title = f"Reverse annealing without pausing and with bias (h=0.1) \n for beta = {beta}"
    plt.title(title)

    color1 = "tab:blue"
    ax1.set_xlabel("Annealing time ($\\mu s$)")
    ax1.set_ylabel("Success probability")
    ax1.plot(list(prob.keys()), list(prob.values()), color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = "tab:red"
    ax2.set_ylabel("Average distance from the ground state")
    ax2.plot(list(dist.keys()), [v for v in dist.values()], color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.show()
    fig.savefig(os.path.join(cwd, "results", "plots", f"{dataset}.png"))
