"""
Utilities to perform Gradient Checking
"""
from functools import reduce

import numpy as np


def to_vector(layers, w_grads, b_grads):
    v_params = np.array([])
    v_grads = np.array([])
    params_shapes = {}

    for layer in layers:
        w, b = layer.get_params()
        params_shapes[("w", layer)] = w.shape
        v_params = np.append(v_params, w.reshape(-1, reduce(lambda x, y: x * y, w.shape)))

        params_shapes[("b", layer)] = b.shape
        v_params = np.append(v_params, b.reshape(-1, reduce(lambda x, y: x * y, b.shape)))

        dw = w_grads[layer]
        v_grads = np.append(v_grads, dw.reshape(-1, reduce(lambda x, y: x * y, dw.shape)))

        db = b_grads[layer]
        v_grads = np.append(v_grads, db.reshape(-1, reduce(lambda x, y: x * y, db.shape)))

    v_params = v_params.reshape(v_params.shape[0], 1)
    v_grads = v_grads.reshape(v_grads.shape[0], 1)

    return v_params, v_grads, params_shapes


def to_dict(layers, v_params, params_shapes):
    curr = 0
    params = {}

    for layer in layers:
        sh = params_shapes[("w", layer)]
        to_take = reduce(lambda x, y: x * y, sh)
        w = v_params[curr:curr+to_take].reshape(*sh)
        layer.w = w
        curr += to_take

        sh = params_shapes[("b", layer)]
        to_take = reduce(lambda x, y: x * y, sh)
        b = v_params[curr:curr+to_take].reshape(*sh)
        layer.b = b
        curr += to_take

    return params


def grad_check(nn, x, y, epsilon=1e-7):
    a_last = nn.forward_prop(x)
    nn.backward_prop(a_last, y)
    v_params, v_grads, params_shapes = to_vector(nn.trainable_layers, nn.w_grads, nn.b_grads)
    n_param = v_params.shape[0]
    J_plus = np.zeros((n_param, 1))
    J_minus = np.zeros((n_param, 1))
    grad_approx = np.zeros((n_param, 1))

    for i in range(n_param):
        v_params_plus = np.copy(v_params)
        v_params_plus[i][0] += epsilon
        nn.params = to_dict(nn.trainable_layers, v_params_plus, params_shapes)
        a_last = nn.forward_prop(x)
        J_plus[i] = nn.compute_cost(a_last, y)

        v_params_minus = np.copy(v_params)
        v_params_minus[i][0] -= epsilon
        nn.params = to_dict(nn.trainable_layers, v_params_minus, params_shapes)
        a_last = nn.forward_prop(x)
        J_minus[i] = nn.compute_cost(a_last, y)

        grad_approx[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    return np.linalg.norm(grad_approx - v_grads) / (np.linalg.norm(v_grads) + np.linalg.norm(grad_approx))
