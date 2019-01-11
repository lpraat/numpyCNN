from functools import reduce

from src.layers.layer import Layer


class Flatten(Layer):
    """Flatten layer.

    Attributes
    ----------
    original_dim : tuple
        Shape of the input ndarray.
    output_dim : tuple
        Shape of the output ndarray.
    """
    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def init(self, in_dim):
        self.original_dim = in_dim
        self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

    def forward(self, a_prev, training):
        return a_prev.reshape(a_prev.shape[0], -1)

    def backward(self, da):
        return da.reshape(da.shape[0], *self.original_dim), None, None

    def get_params(self):
        pass

    def update_params(self, dw, db):
        pass

    def get_output_dim(self):
        return self.output_dim

