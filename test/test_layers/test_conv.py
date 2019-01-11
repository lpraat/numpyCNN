import unittest

import numpy as np

from src.layers.conv import Conv


class TestConv(unittest.TestCase):
    def test_conv_forward(self):
        np.random.seed(1)
        a_prev = np.random.randn(10, 4, 4, 3)
        w = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        c = Conv(2, 2, 8)
        c.init(a_prev.shape[1:])
        c.w = w
        c.b = b
        c.n_w = 4
        c.n_h = 4
        c.pad = 2
        z = c.forward(a_prev, False)
        np.testing.assert_almost_equal(np.mean(z), np.array([0.0489952035289]))
        np.testing.assert_almost_equal(z[3, 2, 1], np.array([-0.61490741, -6.7439236, -2.55153897,
                                                             1.75698377, 3.56208902, 0.53036437,
                                                             5.18531798, 8.75898442]))

    def test_conv_backward(self):
        np.random.seed(1)
        a_prev = np.random.randn(10, 4, 4, 3)
        w = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        c = Conv(2, 2, 8)
        c.init(a_prev.shape[1:])
        c.w = w
        c.b = b
        c.n_w = 4
        c.n_h = 4
        c.pad = 2
        z = c.forward(a_prev, True)

        da, dw, db = c.backward(z)
        np.testing.assert_almost_equal(np.mean(da), np.array([1.45243777754]))
        np.testing.assert_almost_equal(np.mean(dw), np.array([0.172699145831]))
        np.testing.assert_almost_equal(np.mean(db), np.array([0.783923256462]))
