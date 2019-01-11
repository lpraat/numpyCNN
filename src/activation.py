import numpy as np


class ActivationFunction:
    def f(self, x):
        raise NotImplementedError

    def df(self, x, cached_y=None):
        raise NotImplementedError


class Identity(ActivationFunction):
    def f(self, x):
        return x

    def df(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(ActivationFunction):
    def f(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def df(self, x, cached_y=None):
        y = cached_y if cached_y is not None else self.f(x)
        return y * (1 - y)


class ReLU(ActivationFunction):
    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


class SoftMax(ActivationFunction):
    def f(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    def df(self, x, cached_y=None):
        raise NotImplementedError


identity = Identity()
sigmoid = Sigmoid()
relu = ReLU()
softmax = SoftMax()
