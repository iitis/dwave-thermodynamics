import pickle


with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\checkpoints\\checkpoint_result_reverse_s0.5_beta1.pkl",
          "rb") as f:
    beta_eff, mean_E, var_E, mean_Q, var_Q = pickle.load(f)


with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\checkpoints\\checkpoint_result_reverse_s0.5_beta1_200.pkl",
          "rb") as f:
    beta_eff_200, mean_E_200, var_E_200, mean_Q_200, var_Q_200 = pickle.load(f)


beta_eff = beta_eff + beta_eff_200
mean_Q = mean_Q + mean_Q_200
var_Q = var_Q + var_Q_200

with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\"
          "result_reverse_s0.5_beta1.pkl",
          "wb") as f:
    pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)