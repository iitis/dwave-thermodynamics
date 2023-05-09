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
    [beta_eff_p, mean_E_p, var_E_r, mean_Q_r, var_Q_p] = pickle.load(f)
with open('results/results_pausing_h_q_varE.pkl', 'rb') as f:
    [var_E_p, mean_Q_p] = pickle.load(f)        



#reverse annealing with h
tau=[2.00,24.00,46.00,68.00,90.00,112.00,134.00,156.00]
h=0.1
eta=[]
for x in tau:
    k=mean_Q_r[(x,h)]/(np.sqrt(np.array(var_Q_r[(x,h)])+np.square(np.array(mean_Q_r[(x,h)]))))
    w=2*k*np.arctanh(k)/beta_eff_r[(x,h)] + (1-1/beta_eff_r[(x,h)])*mean_Q_r[(x,h)]
    q=-2*k*np.arctanh(k)/beta_eff_r[(x,h)] + (1/beta_eff_r[(x,h)])*mean_Q_r[(x,h)]
    eta+=[-w/q]
#reverse pausing with h

# h=0.1
# eta=[]
# for x in tau:
#     k=mean_Q_p[(x,h)]/(np.sqrt(np.array(var_Q_p[(x,h)])+np.square(np.array(mean_Q_p[(x,h)]))))
#     w=2*k*np.arctanh(k)/beta_eff_p[(x,h)] + (1-1/beta_eff_p[(x,h)])*mean_Q_p[(x,h)]
#     q=-2*k*np.arctanh(k)/beta_eff_p[(x,h)] + (1/beta_eff_p[(x,h)])*mean_Q_p[(x,h)]
#     eta+=[-w/q]

plt.figure(num=None)
plt.plot(tau,eta,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$\eta_{th}$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
#plt.savefig("figs/reverse pausing/T=1/efficiency.pdf")
plt.show()



