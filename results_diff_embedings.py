import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams.update({
#     "text.usetex": True,
# })

# with open('results/results_pausing_Q1.pkl', 'rb') as f:
#     [beta_eff_1, mean_E_1, var_E_1, mean_Q_1, var_Q_1] = pickle.load(f)

# with open('results/results_pausing_Q2.pkl', 'rb') as f:
#     [beta_eff_2, mean_E_2, var_E_2, mean_Q_2, var_Q_2] = pickle.load(f)

# with open('results/results_pausing_Q3.pkl', 'rb') as f:
#     [beta_eff_3, mean_E_3, var_E_3, mean_Q_3, var_Q_3] = pickle.load(f)

# with open('results/results_pausing_Q4.pkl', 'rb') as f:
#     [beta_eff_4, mean_E_4, var_E_4, mean_Q_4, var_Q_4] = pickle.load(f)

with open('results/results_reverse_Q1.pkl', 'rb') as f:
    [beta_eff_1, mean_E_1, var_E_1, mean_Q_1, var_Q_1] = pickle.load(f)

with open('results/results_reverse_Q2.pkl', 'rb') as f:
    [beta_eff_2, mean_E_2, var_E_2, mean_Q_2, var_Q_2] = pickle.load(f)

with open('results/results_reverse_Q3.pkl', 'rb') as f:
    [beta_eff_3, mean_E_3, var_E_3, mean_Q_3, var_Q_3] = pickle.load(f)

with open('results/results_reverse_Q4.pkl', 'rb') as f:
    [beta_eff_4, mean_E_4, var_E_4, mean_Q_4, var_Q_4] = pickle.load(f)

num_s_bar = 10

print('Loaded data from <-', f.name)
#mylist1=sorted(beta_eff_1.items()); mylist2=sorted(beta_eff_2.items()); mylist3=sorted(beta_eff_3.items()); mylist4=sorted(beta_eff_4.items())
#mylist1=sorted(mean_Q_1.items()); mylist2=sorted(mean_Q_2.items()); mylist3=sorted(mean_Q_3.items()); mylist4=sorted(mean_Q_4.items())
# mylist1=sorted(mean_E_1.items()); mylist2=sorted(mean_E_2.items()); mylist3=sorted(mean_E_3.items()); mylist4=sorted(mean_E_4.items())

# x1, y1 = zip(*mylist1); x2, y2 = zip(*mylist2); x3, y3 = zip(*mylist3); x4, y4 = zip(*mylist4)

e1=[];e2=[];e3=[];e4=[]
for x in mean_Q_1.keys():
    k1=mean_Q_1[x]/np.sqrt(var_Q_1[x]+np.square(mean_Q_1[x]))
    y1=2*k1*np.arctanh(k1)
    w1=(y1/beta_eff_1[x]) + (1- (1/beta_eff_1[x]))*mean_Q_1[x]
    q1=-(y1/beta_eff_1[x]) + ((1/beta_eff_1[x]))*mean_Q_1[x]

    k2=mean_Q_2[x]/np.sqrt(var_Q_2[x]+np.square(mean_Q_2[x]))
    y2=2*k2*np.arctanh(k2)
    w2=(y2/beta_eff_2[x]) + (1- (1/beta_eff_2[x]))*mean_Q_2[x]
    q2=-(y2/beta_eff_2[x]) + ((1/beta_eff_2[x]))*mean_Q_2[x]

    k3=mean_Q_3[x]/np.sqrt(var_Q_3[x]+np.square(mean_Q_3[x]))
    y3=2*k3*np.arctanh(k3)
    w3=(y3/beta_eff_3[x]) + (1- (1/beta_eff_3[x]))*mean_Q_3[x]
    q3=-(y3/beta_eff_3[x]) + ((1/beta_eff_3[x]))*mean_Q_3[x]

    k4=mean_Q_4[x]/np.sqrt(var_Q_4[x]+np.square(mean_Q_4[x]))
    y4=2*k4*np.arctanh(k4)
    w4=(y4/beta_eff_4[x]) + (1- (1/beta_eff_4[x]))*mean_Q_4[x]
    q4=-(y4/beta_eff_4[x]) + ((1/beta_eff_4[x]))*mean_Q_4[x]

    e1+=[-w1/q1]
    e2+=[-w2/q2]
    e3+=[-w3/q3]
    e4+=[-w4/q4]

plt.plot(mean_Q_1.keys(),e1,'-o',label=r"$Q_1$")
plt.plot(mean_Q_1.keys(),e2,'-*',label=r"$Q_2$")
plt.plot(mean_Q_1.keys(),e3,'-x',label=r"$Q_3$")
plt.plot(mean_Q_1.keys(),e4,'-^',label=r"$Q_4$")

plt.ylabel(r'$\eta$',fontsize=20)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t$ $(\mu s)$',fontsize=20)
plt.xticks(fontsize=12.5)
plt.legend(fontsize=20)
plt.tight_layout
plt.savefig("efficiency_reverse_Q.pdf")

plt.show()

# plt.plot(x1, y1,'-*',label=r"$Q_1$")
# plt.plot(x2, y2,'-.',label=r"$Q_2$")
# plt.plot(x3, y3,'-x',label=r"$Q_3$")
# plt.plot(x4, y4,'-o',label=r"$Q_4$")

# plt.ylabel(r'$\langle  E \rangle $',fontsize=17)
# #plt.ylabel(r'$\beta_2$',fontsize=20)
# #plt.ylabel(r'$\eta $',fontsize=17)
# plt.yticks(fontsize=12.5)
# plt.xlabel(r'$t$ $(\mu s)$',fontsize=20)
# plt.xticks(fontsize=12.5)

# plt.legend(fontsize=20)
# plt.tight_layout()
# plt.savefig("meanE_pausing_Q.pdf")
# plt.show()


