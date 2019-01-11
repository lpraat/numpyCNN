import unittest

import numpy as np

from src.activation import relu, sigmoid
from src.layers.fc import FullyConnected


class TestLayer(unittest.TestCase):

    def test_fully_connected_forward(self):
        np.random.seed(2)
        layer_size = 1
        previous_layer_size = 3
        a_prev = np.random.randn(previous_layer_size, 2)
        w = np.random.randn(layer_size, previous_layer_size)
        b = np.random.randn(layer_size, 1).reshape(1, layer_size)

        fc_sigmoid = FullyConnected(3, sigmoid)
        fc_relu = FullyConnected(3, relu)

        fc_sigmoid.w = w
        fc_sigmoid.b = b
        fc_relu.w = w
        fc_relu.b = b

        np.testing.assert_array_almost_equal(fc_sigmoid.forward(a_prev.T, False), np.array([[0.96890023, 0.11013289]]).T)
        np.testing.assert_array_almost_equal(fc_relu.forward(a_prev.T, False), np.array([[3.43896131, 0.]]).T)






