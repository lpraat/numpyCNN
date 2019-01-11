# NumpyCNN
A simple vectorized implementation of a Convolutional Neural Network in plain numpy I wrote while learning about neural networks, aaaand more.   

##### Example

```python
# ... some imports here ...
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
```


In mnist_cnn.py there is a complete example with a simple model I used to get 99.06% accuracy on the mnist test dataset.

## You can find an implementation of: 
#### Gradient Checking
To check the correctness of derivatives during backpropagation as explained [here](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)  
There are examples of its usage in the tests.

#### Layers
- FullyConnected (Dense)
- Conv (Conv2D)
- Pool (MaxPool2D, AveragePool2D)
- Dropout
- Flatten

#### Optimizers
- Gradient Descent
- RMSProp
- Adam