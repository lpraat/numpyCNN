from functools import reduce

import numpy as np

from src.optimizer import gradient_descent


class NeuralNetwork:
    """Neural network model.

    Attributes
    ----------
    layers : list
        Layers used in the model.
    w_grads : dict
        Weights' gradients during backpropagation.
    b_grads : dict
        Biases' gradients during backpropagation.
    cost_function : CostFunction
        Cost function to be minimized.
    optimizer : Optimizer
        Optimizer used to update trainable parameters (weights and biases).
    l2_lambda : float
        L2 regularization parameter.
    trainable_layers: list
        Trainable layers(those that have trainable parameters) used in the model.
    """

    def __init__(self, input_dim, layers, cost_function, optimizer=gradient_descent, l2_lambda=0):
        self.layers = layers
        self.w_grads = {}
        self.b_grads = {}
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        # Initialize the layers in the model providing the input dimension they should expect
        self.layers[0].init(input_dim)
        for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
            curr_layer.init(prev_layer.get_output_dim())

        self.trainable_layers = set(layer for layer in self.layers if layer.get_params() is not None)
        self.optimizer = optimizer(self.trainable_layers)
        self.optimizer.initialize()

    def forward_prop(self, x, training=True):
        """
        Performs a forward propagation pass.

        Parameters
        ----------
        x : numpy.ndarray
            Input that is fed to the first layer.
        training : bool
            Whether the model is training.

        Returns
        -------
        numpy.ndarray
            Model's output, corresponding to the last layer's activations.
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a, training)

        return a

    def backward_prop(self, a_last, y):
        """
        Performs a backward propagation pass.

        Parameters
        ----------
        a_last : numpy.ndarray
            Last layer's activations.
        y : numpy.ndarray
            Target labels.
        """
        da = self.cost_function.grad(a_last, y)
        batch_size = da.shape[0]

        for layer in reversed(self.layers):
            da_prev, dw, db = layer.backward(da)

            if layer in self.trainable_layers:
                if self.l2_lambda != 0:
                    # Update the weights' gradients also wrt the l2 regularization cost
                    self.w_grads[layer] = dw + (self.l2_lambda / batch_size) * layer.get_params()[0]
                else:
                    self.w_grads[layer] = dw

                self.b_grads[layer] = db

            da = da_prev

    def predict(self, x):
        """
        Calculates the output of the model for the input.

        Parameters
        ----------
        x : numpy.ndarray
            Input.

        Returns
        -------
        numpy.ndarray
            Prediction of the model, corresponding to the last layer's activations.
        """
        a_last = self.forward_prop(x, training=False)
        return a_last

    def update_param(self, learning_rate, step):
        """
        Updates the trainable parameters of the layers in the model.

        Parameters
        ----------
        learning_rate : float
            Update's learning rate.
        step : int
            How many updates have been performed from the start of the training.
        """
        self.optimizer.update(learning_rate, self.w_grads, self.b_grads, step)

    def compute_cost(self, a_last, y):
        """
        Computes the cost, given the output and the target labels.

        Parameters
        ----------
        a_last : numpy.ndarray
            Output.
        y : numpy.ndarray
            Target labels.

        Returns
        -------
        float
            The cost.
        """
        cost = self.cost_function.f(a_last, y)
        if self.l2_lambda != 0:
            batch_size = y.shape[0]
            weights = [layer.get_params()[0] for layer in self.trainable_layers]
            l2_cost = (self.l2_lambda / (2 * batch_size)) * reduce(lambda ws, w: ws + np.sum(np.square(w)), weights, 0)
            return cost + l2_cost
        else:
            return cost

    def train(self, x_train, y_train, mini_batch_size, learning_rate, num_epochs, validation_data):
        """
        Trains the model for a given number of epochs.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training input data.
        y_train : numpy.ndarray
            Training target labels.
        mini_batch_size : int
            Size of a mini batch. Number of samples per parameters update step.
        learning_rate : float
            Parameters' update learning rate.
        num_epochs : int
            The number of epochs.
        validation_data : tuple
            A pair of input data and target labels to evaluate the model on.
        """
        x_val, y_val = validation_data
        print(f"Started training [batch_size={mini_batch_size}, learning_rate={learning_rate}]")
        step = 0
        for e in range(num_epochs):
            print("Epoch " + str(e + 1))
            epoch_cost = 0

            if mini_batch_size == x_train.shape[0]:
                mini_batches = (x_train, y_train)
            else:
                mini_batches = NeuralNetwork.create_mini_batches(x_train, y_train, mini_batch_size)

            num_mini_batches = len(mini_batches)
            for i, mini_batch in enumerate(mini_batches, 1):
                mini_batch_x, mini_batch_y = mini_batch
                step += 1
                epoch_cost += self.train_step(mini_batch_x, mini_batch_y, learning_rate, step) / mini_batch_size
                print("\rProgress {:1.1%}".format(i / num_mini_batches), end="")

            print(f"\nCost after epoch {e+1}: {epoch_cost}")

            print("Computing accuracy on validation set...")
            accuracy = np.sum(np.argmax(self.predict(x_val), axis=1) == y_val) / x_val.shape[0]
            print(f"Accuracy on validation set: {accuracy}")

        print("Finished training")

    def train_step(self, x_train, y_train, learning_rate, step):
        """
        Performs one model training step.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training input data.
        y_train : numpy.ndarray
            Training target labels.
        learning_rate : float
            Parameters' update learning rate.
        step : int
            How many parameters updates have been performed from the start of the training.

        Returns
        -------
        float
            The cost during this training step.
        """
        a_last = self.forward_prop(x_train, training=True)
        self.backward_prop(a_last, y_train)
        cost = self.compute_cost(a_last, y_train)
        self.update_param(learning_rate, step)
        return cost

    @staticmethod
    def create_mini_batches(x, y, mini_batch_size):
        """
        Creates sample mini batches from input and target labels batches.
        x : numpy.ndarray
            Input batch.
        y : numpy.ndarray
            Target labels batch.

        Returns
        -------
        list
            Mini batches pairs of input and target labels.
        """
        batch_size = x.shape[0]
        mini_batches = []

        p = np.random.permutation(x.shape[0])
        x, y = x[p, :], y[p, :]
        num_complete_minibatches = batch_size // mini_batch_size

        for k in range(0, num_complete_minibatches):
            mini_batches.append((
                x[k * mini_batch_size:(k + 1) * mini_batch_size, :],
                y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            ))

        # Fill with remaining data, if needed
        if batch_size % mini_batch_size != 0:
            mini_batches.append((
                x[num_complete_minibatches * mini_batch_size:, :],
                y[num_complete_minibatches * mini_batch_size:, :]
            ))

        return mini_batches
