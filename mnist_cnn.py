import numpy as np

import mnist
from src.activation import relu, softmax
from src.cost import softmax_cross_entropy
from src.layers.conv import Conv
from src.layers.dropout import Dropout
from src.layers.fc import FullyConnected
from src.layers.flatten import Flatten
from src.layers.pool import Pool
from src.nn import NeuralNetwork
from src.optimizer import adam


def one_hot(x, num_classes=10):
    out = np.zeros((x.shape[0], num_classes))
    out[np.arange(x.shape[0]), x[:, 0]] = 1
    return out


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
    y_train = one_hot(y_train.reshape(y_train.shape[0], 1))
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    mnist.init()
    x_train, y_train, x_test, y_test = preprocess(*mnist.load())

    cnn = NeuralNetwork(
        input_dim=(28, 28, 1),
        layers=[
            Conv(5, 1, 32, activation=relu),
            Pool(2, 2, 'max'),
            Dropout(0.75),
            Flatten(),
            FullyConnected(128, relu),
            Dropout(0.9),
            FullyConnected(10, softmax),
        ],
        cost_function=softmax_cross_entropy,
        optimizer=adam
    )

    cnn.train(x_train, y_train,
              mini_batch_size=256,
              learning_rate=0.001,
              num_epochs=30,
              validation_data=(x_test, y_test))
