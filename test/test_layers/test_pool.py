import unittest

import numpy as np

from src.layers.pool import Pool


class TestPool(unittest.TestCase):

    def test_pool_forward(self):
        np.random.seed(1)
        a_prev = np.random.randn(2, 4, 4, 3)

        p = Pool(3, 2, 'max')
        p.init(a_prev.shape[1:])
        a = p.forward(a_prev, False)
        np.testing.assert_almost_equal(a, np.array([[[[1.74481176, 0.86540763, 1.13376944]]],
                                                    [[[1.13162939, 1.51981682, 2.18557541]]]]))

        p = Pool(3, 2, 'average')
        p.init(a_prev.shape[1:])
        a = p.forward(a_prev, False)
        np.testing.assert_almost_equal(a, np.array([[[[0.02105773, -0.20328806, -0.40389855]]],
                                                    [[[-0.22154621, 0.51716526, 0.48155844]]]]))

    def test_pool_backward(self):
        np.random.seed(1)
        a_prev = np.random.randn(5, 5, 3, 2)
        p = Pool(2, 1, 'max')
        p.init(a_prev.shape[1:])
        p.forward(a_prev, True)
        da = np.random.randn(5, 4, 2, 2)
        da_prev, _, _ = p.backward(da)
        np.testing.assert_almost_equal(np.mean(da), np.array([0.145713902729]))
        np.testing.assert_almost_equal(da_prev[1, 1], np.array([[0., 0.],
                                                                [5.05844394, -1.68282702],
                                                                [0., 0.]]))

        np.random.seed(1)
        a_prev = np.random.randn(5, 5, 3, 2)
        p = Pool(2, 1, 'average')
        p.init(a_prev.shape[1:])
        p.forward(a_prev, True)
        da = np.random.randn(5, 4, 2, 2)
        da_prev, _, _ = p.backward(da)
        np.testing.assert_almost_equal(np.mean(da), np.array([0.145713902729]))
        np.testing.assert_array_almost_equal(da_prev[1, 1], np.array([[0.08485462, 0.2787552],
                                                                      [1.26461098, -0.25749373],
                                                                      [1.17975636, -0.53624893]]))
