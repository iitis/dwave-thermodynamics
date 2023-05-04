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

num_s_bar = 10

#mean_E_therm = mean_E_therm[10:20]

print('Loaded data from <-', f.name)

# with open("checkpoint.pkl", "rb") as f:
#     mean_E_therm_long, var_E_therm_long = pickle.load(f)

x = np.linspace(1,200,num = 10)
y = np.array(mean_E)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.ylabel(r'$\langle  E_1 \rangle $',fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$t$ $(\mu s)$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()
#
x = np.linspace(1,200 ,num = num_s_bar)
y = np.array(mean_E)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$\langle \Delta E_1 \rangle$',fontsize=20)
plt.xlabel(r'$\bar{s}$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()



x = np.linspace(1, 200,num = num_s_bar)
y = np.array(var_E)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'$var(\Delta E_1)$',fontsize=20)
plt.xlabel(r'$\bar{s}$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()



x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
y = 2*k*np.arctanh(k)

plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle \sigma \rangle$',fontsize=20)
plt.xlabel(r'$\bar{s}$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()



x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
y=2*k*np.arctanh(k)
z=(-np.array(mean_Q)-k/3.25)

plt.figure(num=None, figsize=(10, 7))
plt.plot(x,z,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle W \rangle$',fontsize=20)
plt.xlabel(r'$\bar{s}$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()

x = np.linspace(1,200,num = num_s_bar)
chain_lenght = 300
k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
y=2*k*np.arctanh(k)/3.25

plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.yticks(fontsize=20)
plt.ylabel(r'Lower bound to $\langle \Delta E_2 \rangle$',fontsize=20)
plt.xlabel(r'$\bar{s}$',fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
plt.show()

