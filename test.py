import pickle

with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\checkpoints\\checkpoints_h1.0_pausing.pkl",
          "rb") as f:
    beta_eff_pause, mean_E_pause, var_E_pause, mean_Q_pause, var_Q_pause = pickle.load(f)
    print(mean_E_pause)