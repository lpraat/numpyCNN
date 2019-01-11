import unittest

import numpy as np

from src.layers.flatten import Flatten


class TestFlatten(unittest.TestCase):

    def test_flatten(self):
        batch_size = 10
        n_h, n_w, n_c = 32, 32, 3
        a_prev = np.random.randn(batch_size, n_h, n_w, n_c)
        f = Flatten()
        f.init((n_h, n_w, n_c))
        self.assertEqual(f.get_output_dim(), n_h * n_w * n_c)
        self.assertTupleEqual(f.forward(a_prev, False).shape, (batch_size, n_h * n_w * n_c))
        da, _, _ = f.backward(a_prev)
        self.assertTupleEqual(da.shape, (batch_size, n_h, n_w, n_c))
        np.testing.assert_array_almost_equal(a_prev, da)
