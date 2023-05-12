import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

with open('results/results_reverse_Q1.pkl', 'rb') as f:
    [beta_eff_1, mean_E_1, var_E_1, mean_Q_1, var_Q_1] = pickle.load(f)

with open('results/results_reverse_Q2.pkl', 'rb') as f:
    [beta_eff_2, mean_E_2, var_E_2, mean_Q_2, var_Q_2] = pickle.load(f)

with open('results/results_reverse_Q3.pkl', 'rb') as f:
    [beta_eff_3, mean_E_3, var_E_3, mean_Q_3, var_Q_3] = pickle.load(f)

with open('results/results_reverse_Q4.pkl', 'rb') as f:
    [beta_eff_4, mean_E_4, var_E_4, mean_Q_4, var_Q_4] = pickle.load(f)

with open('results/results_eff_Q1_reverse.pkl', "rb") as f:
        prob1, dist1 = pickle.load(f)
with open('results/results_eff_Q2_reverse.pkl', "rb") as f:
        prob2, dist2 = pickle.load(f)
with open('results/results_eff_Q3_reverse.pkl', "rb") as f:
        prob3, dist3 = pickle.load(f)
with open('results/results_eff_Q4_reverse.pkl', "rb") as f:
        prob4, dist4 = pickle.load(f)

print('Loaded data from <-', f.name)




w=[];q=[];e=[];
for x in mean_Q_4.keys():
    k1=mean_Q_4[x]/np.sqrt(var_Q_3[x]+np.square(mean_Q_4[x]))
    y1=2*k1*np.arctanh(k1)
    w+=[(y1/beta_eff_4[x]) + (1- (1/beta_eff_4[x]))*mean_Q_4[x]]
    q+=[-(y1/beta_eff_4[x]) + ((1/beta_eff_4[x]))*mean_Q_4[x]]
e=[-w[x]/q[x] for x in range(10)]


eta_comp=[list(prob4.values())[x]/w[x] for x in range(10) ]

fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.set_xlabel(r"$t (\mu s)$", fontsize=20)
ax1.tick_params(axis="x", labelsize=20)
ax1.set_ylabel(r"$\eta_{th}$", fontsize=20)
ax1.plot(mean_Q_4.keys(), e, '-o',color=color1)
ax1.tick_params(axis="y", labelcolor=color1, labelsize=20)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color2 = "tab:red"
ax2.set_ylabel(r"$\eta_{comp}$", fontsize=20)
ax2.plot(mean_Q_4.keys(), eta_comp, '-x',color=color2)
ax2.tick_params(axis="y", labelcolor=color2, labelsize=20)
plt.tight_layout()
plt.savefig("figs/nthVSncomp_Q4_reverse.pdf")
plt.show()