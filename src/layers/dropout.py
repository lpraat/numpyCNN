import numpy as np

from src.layers.layer import Layer


class Dropout(Layer):
    """Dropout layer.

    Attributes
    ----------
    keep_prob : float
        Probability that a neuron is kept.
    mask_dim : tuple
        Shape of the input ndarray.
    cached_mask : numpy.ndarray
        Mask representing kept/dropped neurons.
    """
    def __init__(self, keep_prob):
        super().__init__()
        assert 0 < keep_prob < 1, "Keep probability must be between 0 and 1"
        self.keep_prob = keep_prob
        self.mask_dim = None
        self.cached_mask = None

    def init(self, in_dim):
        self.mask_dim = in_dim

    def forward(self, a_prev, training):
        if training:
            mask = (np.random.rand(*a_prev.shape) < self.keep_prob)
            a = self.inverted_dropout(a_prev, mask)

            # Cache for backward pass
            self.cached_mask = mask

            return a

        return a_prev

    def backward(self, da):
        return self.inverted_dropout(da, self.cached_mask), None, None

    def update_params(self, dw, db):
        pass

    def get_params(self):
        pass

    def get_output_dim(self):
        return self.mask_dim

    def inverted_dropout(self, a, mask):
        a *= mask
        a /= self.keep_prob
        return a

