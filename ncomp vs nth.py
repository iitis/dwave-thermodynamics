import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.weight'] = 'normal'
# plt.rcParams['mathtext.fontset'] = 'stix'

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",       # LaTeX will use Computer Modern
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}",  # optional
})


def calculate_eta(beta_eff, mean_E, var_E, mean_Q, var_Q):
    eta = []
    w = []
    q = []
    k = mean_Q / (np.sqrt(np.array(var_Q) + np.square(np.array(mean_Q))))
    w = [2 * k[x] * np.arctanh(k[x]) / beta_eff[x] + (1 - 1 / beta_eff[x]) * mean_Q[x] for x in range(10)]
    q = [-2 * k[x] * np.arctanh(k[x]) / beta_eff[x] + (1 / beta_eff[x]) * mean_Q[x] for x in range(10)]

    eta = [-w[x] / q[x] for x in range(10)]
    #eta_comp = [list(prob.values())[x] / w[x] for x in range(10)]
    return eta

def calculate_etas_h(beta_eff, mean_E, var_E, mean_Q, var_Q):
    sol = []
    tau = ['2.00', '24.00', '46.00', '68.00', '90.00', '112.00', '134.00', '156.00']
    t = [2.00, 24.00, 46.00, 68.00, 90.00, 112.00, 134.00, 156.00]
    for h in [0.1, 0.5, 1.0]:
        mean_Q_p = [mean_Q[(x, h)] for x in t]
        var_Q_p = [var_Q[(x, h)] for x in t]
        beta_eff_p = [beta_eff[(x, h)] for x in t]

        eta = []
        w = []
        q = []
        for x in range(8):
            k = mean_Q_p[x] / (np.sqrt(np.array(var_Q_p[x]) + np.square(np.array(mean_Q_p[x]))))
            w += [2 * k * np.arctanh(k) / beta_eff_p[x] + (1 - 1 / beta_eff_p[x]) * mean_Q_p[x]]
            q += [-2 * k * np.arctanh(k) / beta_eff_p[x] + (1 / beta_eff_p[x]) * mean_Q_p[x]]
        eta = [-w[x] / q[x] for x in range(8)]
        sol.append(eta)
    return sol

#pausing
# k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
# w=[2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1-1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]
# q=[-2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]

with open('results/result_reverse_s0.5_beta1.pkl', 'rb') as f:
    [beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)

with open('results/results_reverse_s0.41_h.pkl', 'rb') as f:
    [beta_eff_r, mean_E_r, _, _, var_Q_r] = pickle.load(f)

with open('results/results_pausing_s0.41_h.pkl', 'rb') as f:
    [beta_eff_p, mean_E_p, var_E_r, mean_Q_r, var_Q_p] = pickle.load(f)

with open('results/results_pausing_h_q_varE.pkl', 'rb') as f:
    [var_E_p, mean_Q_p] = pickle.load(f)

# with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_0.1.pkl', 'rb') as f:
#     [beta_eff1, mean_E1, var_E1, mean_Q1, var_Q1] = pickle.load(f)
#
# with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_0.5.pkl', 'rb') as f:
#     [beta_eff2, mean_E2, var_E2, mean_Q2, var_Q2] = pickle.load(f)
#
# with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_1.pkl', 'rb') as f:
#     [beta_eff3, mean_E3, var_E3, mean_Q3, var_Q3] = pickle.load(f)



with open('results/results_eff_reverse_beta_1_s_0.5.pkl', "rb") as f:
        prob, dist = pickle.load(f)

num_s_bar = 10
x = np.linspace(1,200,num = num_s_bar)

if __name__ == '__main__':


    eta = calculate_eta(beta_eff, mean_E, var_E, mean_Q, var_Q)
    [eta1, eta2, eta3] = calculate_etas_h(beta_eff_r, mean_E_r, var_E_r, mean_Q_r, var_Q_r)


    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"total annealing time $[\mu s]$", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.set_ylabel(r"$\eta_{th}$", fontsize=30)
    ax1.plot(x[:8], eta[:8])
    ax1.plot(x[:8], eta1, "--")
    ax1.plot(x[:8], eta2, "-.")
    ax1.plot(x[:8], eta3, ":")

    ymin = 0.85
    ymax = 1
    ax1.set_ylim(ymin-0.01, ymax+0.01)

    ax1.set_yticks(np.linspace(ymin, ymax, 4))
    ax1.tick_params(axis="y", labelsize=20)

    # instantiate a second axes that shares the same x-axis

    # ax2.set_ylabel(r"$\eta_{comp}$", fontsize=20)
    # ax2.plot(x, eta_comp, '-x',color=color2)
    # ax2.tick_params(axis="y", labelcolor=color2, labelsize=20)
    plt.legend([r"$h=0$", r"$h=0.1$", r"$h=0.5$", r"$h=1.0$"], fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.savefig("figs/eta_th_no_pause.pdf")
    plt.show()
