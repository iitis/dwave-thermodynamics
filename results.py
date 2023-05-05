import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'

with open('results/results_pausing_s0.5_beta1.pkl', 'rb') as f:
    [mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)

num_s_bar = 10

#mean_E_therm = mean_E_therm[10:20]

print('Loaded data from <-', f.name)

# with open("checkpoint.pkl", "rb") as f:
#     mean_E_therm_long, var_E_therm_long = pickle.load(f)

x = np.linspace(1,200,num = 10)
y = np.array(mean_E)/2
plt.figure(num=None)
plt.plot(x,y,'o--')
plt.ylabel(r'$\langle  E_1 \rangle $',fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$t$ $(\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/<E1>.pdf")
#plt.show()
#
x = np.linspace(1,200 ,num = num_s_bar)
y = np.array(beta_eff)
plt.figure(num=None)
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$\beta_2$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/beta2.pdf")
#plt.show()



x = np.linspace(1, 200,num = num_s_bar)
y = np.array(var_E)
plt.figure(num=None)
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$var(\Delta E_1)$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/<vardeltaE1>.pdf")
#plt.show()



x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
y = 2*k*np.arctanh(k)

plt.figure(num=None)
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle \sigma \rangle$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/entropyprod.pdf")
#plt.show()



x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
y=[2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1-1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]

plt.figure(num=None)
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle W \rangle$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/<W>.pdf")
#plt.show()

x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
y=[-2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]

plt.figure(num=None)
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle \Delta E_2 \rangle$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/<-Q>.pdf")
#plt.show()

x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/(2*np.sqrt(np.array(var_Q)+np.square(np.array(mean_Q)/2)))
w=[2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1-1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]
q=[-2*k[x]*np.arctanh(k[x])/beta_eff[x] + (1/beta_eff[x])*mean_Q[x]/2 for x in range(10)]
eta=[-w[x]/q[x] for x in range(10)]

plt.figure(num=None)
plt.plot(x,eta,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$\eta_{th}$',fontsize=20)
plt.xlabel(r'$t (\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig("figs/reverse pausing/T=1/efficiency.pdf")
#plt.show()