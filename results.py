import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('results/average_spin_energy_s0.5_beta1.pkl', 'rb') as f:
    [mean_E_therm, var_E_therm] = pickle.load(f) #beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)

num_s_bar = 5

#mean_E_therm = mean_E_therm[10:20]

print('Loaded data from <-', f.name)

# with open("checkpoint.pkl", "rb") as f:
#     mean_E_therm_long, var_E_therm_long = pickle.load(f)

x = np.linspace(1,200,num = 10)
y = np.array(mean_E_therm)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.ylabel(r'$\langle  E_1 \rangle $',fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t$ $(\mu s)$',fontsize=15)
plt.xticks(fontsize=12.5)
plt.show()
#
# x = np.linspace(0,0.9,num = num_s_bar)
# y = np.array(mean_E)
# plt.figure(num=None, figsize=(10, 7))
# plt.plot(x,y,'o--')
# plt.yticks(fontsize=12.5)
# plt.ylabel(r'$\langle \Delta E_1 \rangle$',fontsize=15)
# plt.xlabel(r'$\bar{s}$',fontsize=17)
# plt.xticks(fontsize=12.5)
# plt.show()
#
#
#
# x = np.linspace(0,0.9,num = num_s_bar)
# y = np.array(var_E)
# plt.figure(num=None, figsize=(10, 7))
# plt.plot(x,y,'o--')
# plt.yticks(fontsize=12.5)
# plt.ylabel(r'$var(\Delta E_1)$',fontsize=17)
# plt.xlabel(r'$\bar{s}$',fontsize=17)
# plt.xticks(fontsize=12.5)
# plt.show()
#
#
#
# x = np.linspace(0.1,0.9,num = num_s_bar)
# chain_lenght = 300
# k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
# y = 2*k*np.arctanh(k)
#
# plt.figure(num=None, figsize=(10, 7))
# plt.plot(x,y,'o--')
# plt.yticks(fontsize=12.5)
# plt.ylabel(r'Lower bound to $\langle \sigma \rangle$',fontsize=17)
# plt.xlabel(r'$\bar{s}$',fontsize=17)
# plt.xticks(fontsize=12.5)
# plt.show()
#
#
#
# x = np.linspace(0.1,0.9,num = num_s_bar)
# chain_lenght = 300
# k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
# y=2*k*np.arctanh(k)
# z=(-np.array(mean_Q)-k/3.25)
#
# plt.figure(num=None, figsize=(10, 7))
# plt.plot(x,z,'o--')
# plt.yticks(fontsize=12.5)
# plt.ylabel(r'Lower bound to $\langle W \rangle$',fontsize=17)
# plt.xlabel(r'$\bar{s}$',fontsize=17)
# plt.xticks(fontsize=12.5)
# plt.show()
#
# x = np.linspace(0.1,0.9,num = num_s_bar)
# chain_lenght = 300
# k=mean_Q/np.sqrt(var_Q+np.square(mean_Q))
# y=2*k*np.arctanh(k)/3.25
#
# plt.figure(num=None, figsize=(10, 7))
# plt.plot(x,y,'o--')
# plt.yticks(fontsize=12.5)
# plt.ylabel(r'Lower bound to $\langle \Delta E_2 \rangle$',fontsize=17)
# plt.xlabel(r'$\bar{s}$',fontsize=17)
# plt.xticks(fontsize=12.5)
# plt.show()

