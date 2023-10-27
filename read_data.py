import pickle
import os
import copy

cwd = os.getcwd()

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_2.pkl"), "rb") as f:
    beta_eff_20_60, mean_E_20_60, var_E_20_60, mean_Q_20_60, var_Q_20_60 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo.pkl"), "rb") as f:
    beta_eff_0, mean_E_0, var_E_0, mean_Q_0, var_Q_0 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_0+++.pkl"), "rb") as f:
    beta_eff_0_2, mean_E_0_2, var_E_0_2, mean_Q_0_2, var_Q_0_2 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_0_rest.pkl"), "rb") as f:
    beta_eff_0_3, mean_E_0_3, var_E_0_3, mean_Q_0_3, var_Q_0_3 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_60.pkl"), "rb") as f:
    beta_eff_60, mean_E_60, var_E_60, mean_Q_60, var_Q_60 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_60+.pkl"), "rb") as f:
    beta_eff_60_2, mean_E_60_2, var_E_60_2, mean_Q_60_2, var_Q_60_2 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_80.pkl"), "rb") as f:
    beta_eff_80, mean_E_80, var_E_80, mean_Q_80, var_Q_80 = pickle.load(f)

with open(os.path.join(cwd, "results", "checkpoints", "checkpoints_pegasus_thermo_100.pkl"), "rb") as f:
    beta_eff_100, mean_E_100, var_E_100, mean_Q_100, var_Q_100 = pickle.load(f)

beta_eff = copy.deepcopy(beta_eff_0)
beta_eff.update(beta_eff_0_2)
beta_eff.update(beta_eff_0_3)
beta_eff.update(beta_eff_20_60)
beta_eff.update(beta_eff_60)
beta_eff.update(beta_eff_60_2)
beta_eff.update(beta_eff_100)

mean_E = copy.deepcopy(mean_E_0)
mean_E.update(mean_E_0_2)
mean_E.update(mean_E_0_3)
mean_E.update(mean_E_20_60)
mean_E.update(mean_E_60)
mean_E.update(mean_E_60_2)
mean_E.update(mean_E_100)

var_E = copy.deepcopy(var_E_0)
var_E.update(var_E_0_2)
var_E.update(var_E_0_3)
var_E.update(var_E_20_60)
var_E.update(var_E_60)
var_E.update(var_E_60_2)
var_E.update(var_E_100)

mean_Q = copy.deepcopy(mean_Q_0)
mean_Q.update(mean_Q_0_2)
mean_Q.update(mean_Q_0_3)
mean_Q.update(mean_Q_20_60)
mean_Q.update(mean_Q_60)
mean_Q.update(mean_Q_60_2)
mean_Q.update(mean_Q_100)

var_Q = copy.deepcopy(var_Q_0)
var_Q.update(var_Q_0_2)
var_Q.update(var_Q_0_3)
var_Q.update(var_Q_20_60)
var_Q.update(var_Q_60)
var_Q.update(var_Q_60_2)
var_Q.update(var_Q_100)

print(list(beta_eff.keys()))
with open(os.path.join(cwd, "results", "pegasus_thermo_all.pkl"), "wb") as f:
    pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)
