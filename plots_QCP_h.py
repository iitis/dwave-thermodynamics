import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

with open('results/results_reverse_s0.41_h.pkl', 'rb') as f:
    [beta_eff_r, mean_E_r, var_E, mean_Q, var_Q_r] = pickle.load(f)
with open('results/results_pausing_s0.41_h.pkl', 'rb') as f:
    [beta_eff, mean_E_p, var_E_r, mean_Q_r, var_Q] = pickle.load(f)
with open('results/results_pausing_h_q_varE.pkl', 'rb') as f:
    [var_E, mean_Q] = pickle.load(f)
with open('results/results_eff_pausing_with_h_beta_1_s_0.41_h_1.pkl', 'rb') as f:
    prob, dist = pickle.load(f)

h=1.0
tau=['2.00','24.00','46.00','68.00','90.00','112.00','134.00','156.00']
t=[2.00,24.00,46.00,68.00,90.00,112.00,134.00,156.00]
mean_Q_p=[mean_Q[(x,h)] for x in tau]
var_Q_p=[var_Q[(x,h)] for x in t]
beta_eff_p=[beta_eff[(x,h)]for x in t]


#reverse annealing with h

# h=1.0
# w=[]
# q=[]
# for x in tau:
#     k=mean_Q_r[(x,h)]/(np.sqrt(np.array(var_Q_r[(x,h)])+np.square(np.array(mean_Q_r[(x,h)]))))
#     w+=[2*k*np.arctanh(k)/beta_eff_r[(x,h)] + (1-1/beta_eff_r[(x,h)])*mean_Q_r[(x,h)]]
#     q+=[-2*k*np.arctanh(k)/beta_eff_r[(x,h)] + (1/beta_eff_r[(x,h)])*mean_Q_r[(x,h)]]

#reverse pausing with h

# h=0.1
eta=[]; w=[]; q=[]
for x in range(8):
    k=mean_Q_p[x]/(np.sqrt(np.array(var_Q_p[x])+np.square(np.array(mean_Q_p[x]))))
    w+=[2*k*np.arctanh(k)/beta_eff_p[x] + (1-1/beta_eff_p[x])*mean_Q_p[x]]
    q+=[-2*k*np.arctanh(k)/beta_eff_p[x] + (1/beta_eff_p[x])*mean_Q_p[x]]
eta=[-w[x]/q[x] for x in range(8)]
eta_comp=[list(prob.values())[x]/w[x] for x in range(8)]

fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.set_xlabel(r"$t (\mu s)$", fontsize=20)
ax1.tick_params(axis="x", labelsize=20)
ax1.set_ylabel(r"$\eta_{th}$", fontsize=20)
#ax1.set_ylabel(r"$\mathcal{P}_{GS}$", fontsize=20)
ax1.plot(t, eta, '-o',color=color1)
#ax1.plot(list(prob.keys()), list(prob.values()), color=color1)
ax1.tick_params(axis="y", labelcolor=color1, labelsize=20)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color2 = "tab:red"
ax2.set_ylabel(r"$\eta_{comp}$", fontsize=20)
#ax2.set_ylabel(r"$\mathcal{F}_{GS}$", fontsize=20)
ax2.plot(t, eta_comp, '-x',color=color2)
#ax2.plot(list(dist.keys()), [v for v in dist.values()], color=color2)
ax2.tick_params(axis="y", labelcolor=color2, labelsize=20)
plt.tight_layout()
plt.savefig("figs/nthVSncomp_pausing_qcp_h=%s.pdf"%h)
#plt.savefig("figs/PF_qcp_pausing_h1.0.pdf")
plt.show()



# # plt.figure(num=None)
# # plt.plot(tau,eta,'o--')
# # plt.yticks(fontsize=20)
# # plt.ylabel(r'$\eta_{th}$',fontsize=20)
# # plt.xlabel(r'$t (\mu s)$',fontsize=20)
# # plt.xticks(fontsize=20)
# # plt.tight_layout()
# # #plt.savefig("figs/reverse pausing/T=1/efficiency.pdf")
# # plt.show()



