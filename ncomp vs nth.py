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


def calculate_eta(beta_eff, mean_E, var_E, mean_Q, var_Q, prob):
    eta = []
    w = []
    q = []
    k = mean_Q / (np.sqrt(np.array(var_Q) + np.square(np.array(mean_Q))))
    w = [2 * k[x] * np.arctanh(k[x]) / beta_eff[x] + (1 - 1 / beta_eff[x]) * mean_Q[x] for x in range(10)]
    q = [-2 * k[x] * np.arctanh(k[x]) / beta_eff[x] + (1 / beta_eff[x]) * mean_Q[x] for x in range(10)]

    eta = [-w[x] / q[x] for x in range(10)]
    eta_comp = [list(prob.values())[x] / w[x] for x in range(10)]
    return eta, eta_comp


def calculate_etas_h(beta_eff, mean_E, var_E, mean_Q, var_Q, prob_h):
    sol = []
    sol2 = []
    tau = ['2.00', '24.00', '46.00', '68.00', '90.00', '112.00', '134.00', '156.00']
    t = [2.00, 24.00, 46.00, 68.00, 90.00, 112.00, 134.00, 156.00]
    for h in [0.1, 0.5, 1.0]:
        mean_Q_p = [mean_Q[(x, h)] for x in t]
        var_Q_p = [var_Q[(x, h)] for x in t]
        beta_eff_p = [beta_eff[(x, h)] for x in t]
        prob = prob_h[h]

        eta = []
        w = []
        q = []
        for x in range(8):
            k = mean_Q_p[x] / (np.sqrt(np.array(var_Q_p[x]) + np.square(np.array(mean_Q_p[x]))))
            w += [2 * k * np.arctanh(k) / beta_eff_p[x] + (1 - 1 / beta_eff_p[x]) * mean_Q_p[x]]
            q += [-2 * k * np.arctanh(k) / beta_eff_p[x] + (1 / beta_eff_p[x]) * mean_Q_p[x]]
        eta = [-w[x] / q[x] for x in range(8)]
        eta_comp = [list(prob.values())[x] / w[x] for x in range(8)]
        sol.append(eta)
        sol2.append(eta_comp)
    return sol, sol2

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

with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_0.1.pkl', 'rb') as f:
    prob1, dist1 = pickle.load(f)

with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_0.5.pkl', 'rb') as f:
    prob2, dist2 = pickle.load(f)

with open('results/results_eff_reverse_with_h_beta_1_s_0.41_h_1.pkl', 'rb') as f:
    prob3, dist3 = pickle.load(f)


with open('results/results_eff_reverse_beta_1_s_0.5.pkl', "rb") as f:
        prob, dist = pickle.load(f)

num_s_bar = 10
x = np.linspace(1,200,num = num_s_bar)

if __name__ == '__main__':

    X_FONT_SIZE = 25
    X_LABEL_SIZE = 20
    Y_FONT_SIZE = 30
    Y_LABEL_SIZE = 20
    LEGEND_FONT_SIZE = 18


    eta, eta_comp = calculate_eta(beta_eff, mean_E, var_E, mean_Q, var_Q, prob)
    prob_h = {0.1: prob1, 0.5: prob2, 1.0: prob3}
    [eta1, eta2, eta3], [comp1, comp2, comp3] = calculate_etas_h(beta_eff_r, mean_E_r, var_E_r, mean_Q_r, var_Q_r, prob_h)


    fig1, ax1 = plt.subplots()

    ax1.set_xlabel(r"total annealing time $[\mu s]$", fontsize=X_FONT_SIZE)
    ax1.tick_params(axis="x", labelsize=X_LABEL_SIZE)
    ax1.tick_params(axis="y", labelsize=Y_LABEL_SIZE)
    ax1.set_ylabel(r"$\eta_{\text{th}}$", fontsize=Y_FONT_SIZE)
    ax1.plot(x[:8], eta[:8])
    ax1.plot(x[:8], eta1, "--")
    ax1.plot(x[:8], eta2, "-.")
    ax1.plot(x[:8], eta3, ":")

    ymin = 0.85
    ymax = 1
    ax1.set_ylim(ymin-0.01, ymax+0.01)
    ax1.set_yticks(np.linspace(ymin, ymax, 4))

    ax1.legend([r"$h=0$", r"$h=0.1$", r"$h=0.5$", r"$h=1.0$"], fontsize=LEGEND_FONT_SIZE)
    fig1.tight_layout()
    ax1.grid()
    fig1.savefig("figs/eta_th_no_pause.pdf")
    plt.show()


    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r"total annealing time $[\mu s]$", fontsize=X_FONT_SIZE)
    ax2.tick_params(axis="x", labelsize=X_LABEL_SIZE)
    ax2.set_ylabel(r"$\eta_{\text{comp}}$", fontsize=Y_FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=Y_LABEL_SIZE)
    ax2.plot(x[:8], eta_comp[:8])
    ax2.plot(x[:8], comp1, "--")
    ax2.plot(x[:8], comp2, "-.")
    ax2.plot(x[:8], comp3, ":")
    fig2.tight_layout()
    ax2.grid()
    ax2.legend([r"$h=0$", r"$h=0.1$", r"$h=0.5$", r"$h=1.0$"], fontsize=LEGEND_FONT_SIZE)
    fig2.savefig("figs/eta_comp_no_pause.pdf")
    plt.show()


    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r"total annealing time $[\mu s]$", fontsize=X_FONT_SIZE)
    ax3.tick_params(axis="x", labelsize=X_LABEL_SIZE)
    ax3.set_ylabel(r"$P_{GS}$", fontsize=Y_FONT_SIZE)
    ax3.tick_params(axis="y", labelsize=Y_LABEL_SIZE)
    ax3.plot(x[:8], list(prob.values())[:8])
    ax3.plot(x[:8], list(prob1.values()), "--")
    ax3.plot(x[:8], list(prob2.values()), "-.")
    ax3.plot(x[:8], list(prob3.values()), ":")
    fig3.tight_layout()
    ax3.grid()
    ax3.legend([r"$h=0$", r"$h=0.1$", r"$h=0.5$", r"$h=1.0$"], fontsize=LEGEND_FONT_SIZE)
    fig3.savefig("figs/prob_no_pause.pdf")
    plt.show()


    fig4, ax4 = plt.subplots()
    ax4.set_xlabel(r"total annealing time $[\mu s]$", fontsize=X_FONT_SIZE)
    ax4.tick_params(axis="x", labelsize=X_LABEL_SIZE)
    ax4.set_ylabel(r"$Q_{GS}$", fontsize=Y_FONT_SIZE)
    ax4.tick_params(axis="y", labelsize=Y_LABEL_SIZE)
    ax4.plot(x[:8], list(dist.values())[:8])
    ax4.plot(x[:8], list(dist1.values()), "--")
    ax4.plot(x[:8], list(dist2.values()), "-.")
    ax4.plot(x[:8], list(dist3.values()), ":")
    fig4.tight_layout()
    ax4.grid()
    ax4.legend([r"$h=0$", r"$h=0.1$", r"$h=0.5$", r"$h=1.0$"], fontsize=LEGEND_FONT_SIZE)
    fig4.savefig("figs/quality_no_pause.pdf")
    plt.show()