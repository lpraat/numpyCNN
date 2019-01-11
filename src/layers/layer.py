class Layer:

    def init(self, in_dim):
        """
        Initializes the layer.

        Parameters
        ----------
        in_dim : int or tuple
            Shape of the input data.
        """
        raise NotImplementedError

    def forward(self, a_prev, training):
        """
        Propagates forward the activations.

        Parameters
        ----------
        a_prev : numpy.ndarray
            The input to this layer which corresponds to the previous layer's activations.
        training : bool
            Whether the model in which this layer is in is training.

        Returns
        -------
        numpy.ndarray
            The activations(output) of this layer.
        """
        raise NotImplementedError

    def backward(self, da):
        """
        Propagates back the gradients.

        Parameters
        ----------
        da : numpy.ndarray
            The gradients wrt the cost of this layer activations.

        Returns
        -------
        tuple
            Triplet with gradients wrt the cost of: previous layer's activations, weights and biases of this layer.
        """
        raise NotImplementedError

    def update_params(self, dw, db):
        """
        Updates parameters given their gradients.

        Parameters
        ----------
        dw : numpy.ndarray
            The gradients wrt the cost of this layer's weights.
        db : numpy.ndarray
            The gradients wrt the cost of this layer's biases.
        """
        raise NotImplementedError

    def get_params(self):
        """
        Returns
        -------
        tuple
            Trainable parameters(weights and biases) of this layer.
        """
        raise NotImplementedError

    def get_output_dim(self):
        """
        Returns
        -------
        tuple
            Shape of the ndarray layer's output.
        """
        raise NotImplementedError
