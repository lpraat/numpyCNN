import numpy as np

from src.layers.layer import Layer


class Pool(Layer):
    """2D pooling layer.

    Supports both max and average pooling.

    Attributes
    ----------
    pool_size : int
        Height and Width of the 2D pooling window.
    stride : int
        Stride along height and width of the input volume on which the pooling operation is applied.
    n_h : int
        Height of the output volume.
    n_w : int
        Width of the output volume.
    n_c : int
        Number of channels of the output volume.
    n_h_prev : int
        Height of the input volume.
    n_w_prev : int
        Width of the input volume.
    n_c_prev : int
        Number of channels of the input volume.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    mode : str
        Pooling mode, either max or average.
    cache : dict
        Cache.
    """
    def __init__(self, pool_size, stride, mode):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.n_h, self.n_w, self.n_c = None, None, None
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
        self.w = None
        self.b = None
        self.mode = mode
        self.cache = {}

    def init(self, in_dim):
        self.n_h_prev, self.n_w_prev, self.n_c_prev = in_dim
        self.n_h = int((self.n_h_prev - self.pool_size) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.pool_size) / self.stride + 1)
        self.n_c = self.n_c_prev

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        # Pool
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                if self.mode == 'max':
                    a_prev_slice = a_prev[:, v_start:v_end, h_start:h_end, :]

                    if training:
                        # Cache for backward pass
                        self.cache_max_mask(a_prev_slice, (i, j))

                    a[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))

                elif self.mode == 'average':
                    a[:, i, j, :] = np.mean(a_prev[:, v_start:v_end, h_start:h_end, :], axis=(1, 2))

                else:
                    raise NotImplementedError("Invalid type of pooling")

        if training:
            self.cache['a_prev'] = a_prev

        return a

    def backward(self, da):
        a_prev = self.cache['a_prev']
        batch_size = a_prev.shape[0]
        da_prev = np.zeros((batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))

        # 'Pool' back
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                if self.mode == 'max':
                    da_prev[:, v_start:v_end, h_start:h_end, :] += da[:, i:i+1, j:j+1, :] * self.cache[(i, j)]

                elif self.mode == 'average':
                    # Distribute the average value back
                    mean_value = np.copy(da[:, i:i+1, j:j+1, :])
                    mean_value[:, :, :, np.arange(mean_value.shape[-1])] /= (self.pool_size * self.pool_size)
                    da_prev[:, v_start:v_end, h_start:h_end, :] += mean_value

                else:
                    raise NotImplementedError("Invalid type of pooling")

        return da_prev, None, None

    def cache_max_mask(self, x, ij):
        mask = np.zeros_like(x)

        # This would be like doing idx = np.argmax(x, axis=(1,2)) if that was possible
        reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        idx = np.argmax(reshaped_x, axis=1)

        ax1, ax2 = np.indices((x.shape[0], x.shape[3]))
        mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1
        self.cache[ij] = mask

    def update_params(self, dw, db):
        pass

    def get_params(self):
        pass

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c


