import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

with open('results/result_reverse_s0.5_beta1.pkl', 'rb') as f:
    [beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)
with open('results/results_eff_reverse_beta_1_s_0.5.pkl', "rb") as f:
        prob, dist = pickle.load(f)

# with open('results/results_pausing_s0.5_beta1.pkl', 'rb') as f:
#      [mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)
# with open('results/results_eff_pausing_beta_1_s_0.5.pkl', "rb") as f:
#          prob, dist = pickle.load(f)

num_s_bar = 10
x = np.linspace(1,200,num = num_s_bar)

#pausing
# k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
# w=[2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1-1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]
# q=[-2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]

# #reverse
k=mean_Q/(np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q))))
w=[2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1-1/beta_eff[x])*mean_Q[x] for x in range(10)]
q=[-2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1/beta_eff[x])*mean_Q[x] for x in range(10)]

eta=[-w[x]/q[x] for x in range(10)]
eta_comp=[list(prob.values())[x]/w[x] for x in range(10) ]

fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.set_xlabel(r"$t (\mu s)$", fontsize=20)
ax1.tick_params(axis="x", labelsize=20)
ax1.set_ylabel(r"$\eta_{th}$", fontsize=20)
ax1.plot(x, eta, '-o',color=color1)
ax1.tick_params(axis="y", labelcolor=color1, labelsize=20)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color2 = "tab:red"
ax2.set_ylabel(r"$\eta_{comp}$", fontsize=20)
ax2.plot(x, eta_comp, '-x',color=color2)
ax2.tick_params(axis="y", labelcolor=color2, labelsize=20)
plt.tight_layout()
plt.savefig("figs/nthVSncomp_reverse.pdf")
plt.show()
