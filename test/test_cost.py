import unittest

import numpy as np

from src.cost import sigmoid_cross_entropy


class TestCost(unittest.TestCase):
    def test_cross_entropy(self):
        y = np.asarray([[1, 1, 1]])
        a_last = np.array([[.8, .9, 0.4]])
        self.assertAlmostEqual(sigmoid_cross_entropy.f(a_last.T, y.T), 0.41493159961539694)
