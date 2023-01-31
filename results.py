import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('experimental_results_1.pkl', 'rb') as f:
    [mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q] = pickle.load(f)

print('Loaded data from <-', f.name)


x = np.linspace(1,200,num = 10)
y = np.array(mean_E_therm)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.ylabel(r'$\langle \Delta E_1 \rangle $',fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t$ $(\mu s)$',fontsize=15)
plt.xticks(fontsize=12.5)
plt.show()

x = np.linspace(0,0.9,num = 9)
y = np.array(mean_E)
plt.figure(num=None, figsize=(10, 7))
plt.plot(x,y,'o--')
plt.yticks(fontsize=12.5)
plt.ylabel(r'$\langle \Delta E_1 \rangle$',fontsize=15)
plt.xlabel(r'$\bar{s}$',fontsize=17)
plt.xticks(fontsize=12.5)
plt.show()

