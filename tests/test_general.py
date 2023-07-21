import unittest
from utils import energy, vectorize
import numpy as np

rng = np.random.default_rng()


class TestGeneralFunctions(unittest.TestCase):
    def test_energy(self):
        chain_length = 300
        h = {i: 0 for i in range(chain_length)}
        J = {(i, i + 1): 1.0 for i in range(chain_length - 1)}
        h_vect, J_vect = vectorize(h, J)
        spins_vect = rng.choice([-1, 1], size=chain_length)
        spins = dict(enumerate(spins_vect.tolist()))
        energy_vect = energy(spins_vect, h_vect, J_vect)

        linear = 0
        quadratic = 0
        for i in h.keys():
            linear += spins[i]*h[i]
        for v, w in J.keys():
            quadratic += spins[v]*spins[w]*J[(v, w)]
        energy_naive = linear + quadratic

        self.assertAlmostEqual(energy_vect, energy_naive)  # add assertion here


if __name__ == '__main__':
    unittest.main()
