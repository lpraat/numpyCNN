import numpy as np

from src.activation import SoftMax
from src.layers.layer import Layer


class FullyConnected(Layer):
    """Densely connected layer.

    Attributes
    ----------
    size : int
        Number of neurons.
    activation : Activation
        Neurons' activation's function.
    is_softmax : bool
        Whether or not the activation is softmax.
    cache : dict
        Cache.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    """
    def __init__(self, size, activation):
        super().__init__()
        self.size = size
        self.activation = activation
        self.is_softmax = isinstance(self.activation, SoftMax)
        self.cache = {}
        self.w = None
        self.b = None

    def init(self, in_dim):
        # He initialization
        self.w = np.random.randn(self.size, in_dim) * np.sqrt(2 / in_dim)

        self.b = np.zeros((1, self.size))

    def forward(self, a_prev, training):
        z = np.dot(a_prev, self.w.T) + self.b
        a = self.activation.f(z)

        if training:
            # Cache for backward pass
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        batch_size = a_prev.shape[0]

        if self.is_softmax:
            # Get back y from the gradient wrt the cost of this layer's activations
            # That is get back y from - y/a = da
            y = da * (-a)

            dz = a - y
        else:
            dz = da * self.activation.df(z, cached_y=a)

        dw = 1 / batch_size * np.dot(dz.T, a_prev)
        db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
        da_prev = np.dot(dz, self.w)

        return da_prev, dw, db

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.size
