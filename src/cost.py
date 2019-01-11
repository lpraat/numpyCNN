import numpy as np


class CostFunction:
    def f(self, a_last, y):
        raise NotImplementedError

    def grad(self, a_last, y):
        raise NotImplementedError


class SigmoidCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        # It would be better to have the logits and use this instead
        # max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        a_last = np.clip(a_last, 1e-10, 1.0 - 1e-10)
        cost = -1 / batch_size * (y * np.log(a_last) + (1 - y) * np.log(1 - a_last)).sum()
        return cost

    def grad(self, a_last, y):
        return - (np.divide(y, a_last) - np.divide(1 - y, 1 - a_last))


class SoftmaxCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        cost = -1 / batch_size * (y * np.log(np.clip(a_last, 1e-10, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        return - np.divide(y, a_last)


softmax_cross_entropy = SoftmaxCrossEntropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()
