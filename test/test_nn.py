import unittest

import numpy as np

from src.activation import relu, softmax, sigmoid
from src.cost import sigmoid_cross_entropy, softmax_cross_entropy
from src.layers.conv import Conv
from src.layers.dropout import Dropout
from src.layers.fc import FullyConnected
from src.layers.flatten import Flatten
from src.layers.pool import Pool
from src.nn import NeuralNetwork
from test.utils.grad_check import grad_check


class TestNn(unittest.TestCase):

    def test_cnn(self):
        x = np.random.randn(2, 8, 8, 3)

        nn = NeuralNetwork(
            input_dim=(8, 8, 3),
            layers=[
                Conv(2, 2, 6, activation=relu),
                Pool(2, 2, 'max'),
                Flatten(),
                FullyConnected(12, relu),
                FullyConnected(6, relu),
                FullyConnected(3, softmax)
            ],
            cost_function=softmax_cross_entropy
        )

        y = np.array([[0, 1, 0], [1, 0, 0]])
        self.assertTrue(grad_check(nn, x, y) < 2e-7)

    def test_regularized_cnn(self):
        x = np.random.randn(2, 8, 8, 3)

        nn = NeuralNetwork(
            input_dim=(8, 8, 3),
            layers=[
                Conv(2, 2, 6, activation=relu),
                Pool(2, 2, 'max'),
                Flatten(),
                FullyConnected(12, relu),
                FullyConnected(6, relu),
                FullyConnected(2, softmax)
            ],
            cost_function=softmax_cross_entropy,
            l2_lambda=0.015
        )

        y = np.array([[0, 1], [1, 0]])
        self.assertTrue(grad_check(nn, x, y) < 2e-7)

    def test_softmax_backprop(self):
        x = np.random.randn(2, 32)

        nn = NeuralNetwork(
            input_dim=32,
            layers=[
                FullyConnected(16, relu),
                FullyConnected(8, relu),
                FullyConnected(4, softmax)
            ],
            cost_function=softmax_cross_entropy,
        )

        y = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(grad_check(nn, x, y) < 2e-7)

    def test_backprop(self):
        x = np.random.randn(32, 3)
        y = np.array([[1, 1, 0]])

        nn = NeuralNetwork(
            input_dim=32,
            layers=[
                FullyConnected(16, relu),
                FullyConnected(8, sigmoid),
                FullyConnected(1, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy,
        )

        self.assertTrue(grad_check(nn, x.T, y.T) < 2e-7)

    def test_regularized_backprop(self):
        x = np.random.randn(10, 3)
        y = np.array([[1, 1, 0]])

        nn = NeuralNetwork(
            input_dim=10,
            layers=[
                FullyConnected(5, relu),
                FullyConnected(3, sigmoid),
                FullyConnected(1, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy,
            l2_lambda=0.5
        )
        self.assertTrue(grad_check(nn, x.T, y.T) < 2e-7)

    def test_params_shape(self):
        x_num = 3
        h_num = 2
        y_num = 1

        np.random.seed(1)

        nn = NeuralNetwork(
            input_dim=x_num,
            layers=[
                FullyConnected(h_num, relu),
                FullyConnected(y_num, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy
        )

        self.assertEqual(nn.layers[0].w.shape, (h_num, x_num))
        self.assertEqual(nn.layers[0].b.shape, (1, h_num))
        self.assertEqual(nn.layers[1].w.shape, (y_num, h_num))
        self.assertEqual(nn.layers[1].b.shape, (1, y_num))

    def test_forward(self):
        np.random.seed(6)
        x = np.random.randn(5, 4)
        x_num = 5
        h1_num = 4
        h2_num = 3
        y_num = 1

        w1 = np.random.randn(h1_num, x_num)
        b1 = np.random.randn(1, h1_num)
        w2 = np.random.randn(h2_num, h1_num)
        b2 = np.random.randn(1, h2_num)
        w3 = np.random.randn(y_num, h2_num)
        b3 = np.random.randn(1, y_num)

        nn = NeuralNetwork(
            input_dim=x_num,
            layers=[
                FullyConnected(h1_num, relu),
                FullyConnected(h2_num, relu),
                FullyConnected(y_num, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy
        )

        nn.layers[0].w = w1
        nn.layers[0].b = b1
        nn.layers[1].w = w2
        nn.layers[1].b = b2
        nn.layers[2].w = w3
        nn.layers[2].b = b3

        a_last = nn.forward_prop(x.T)
        np.testing.assert_array_almost_equal(a_last, np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]]).T)

    def test_regularized_cost(self):
        np.random.seed(1)
        y = np.array([[1, 1, 0, 1, 0]])
        w1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1).reshape(1, 2)
        w2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1).reshape(1, 3)
        w3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        a3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]]).T

        nn = NeuralNetwork(
            input_dim=3,
            layers=[
                FullyConnected(2, relu),
                FullyConnected(3, sigmoid),
                FullyConnected(1, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy,
            l2_lambda=0.1
        )
        nn.layers[0].w = w1
        nn.layers[0].b = b1
        nn.layers[1].w = w2
        nn.layers[1].b = b2
        nn.layers[2].w = w3
        nn.layers[2].b = b3
        self.assertAlmostEqual(nn.compute_cost(a3, y.T), 1.78648594516)

    def test_dropout(self):
        np.random.seed(1)
        x = np.random.randn(3, 5)
        w1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1).reshape(1, 2)
        w2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1).reshape(1, 3)
        w3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)

        nn = NeuralNetwork(
            input_dim=3,
            layers=[
                FullyConnected(2, relu),
                Dropout(keep_prob=0.7),
                FullyConnected(3, relu),
                Dropout(keep_prob=0.7),
                FullyConnected(1, sigmoid)
            ],
            cost_function=sigmoid_cross_entropy,
        )
        nn.layers[0].w = w1
        nn.layers[0].b = b1
        nn.layers[2].w = w2
        nn.layers[2].b = b2
        nn.layers[4].w = w3
        nn.layers[4].b = b3
        np.random.seed(1)
        a_last = nn.forward_prop(x.T)
        np.testing.assert_array_almost_equal(a_last,
                                             np.array([[0.369747, 0.496834, 0.045651, 0.014469, 0.369747]]).T)
