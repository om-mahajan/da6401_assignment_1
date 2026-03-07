import numpy as np
from .activations import get_activation, get_activation_derivative

class NeuralLayer:
    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier", is_output=False):
        self.is_output = is_output
        self.activation_name = activation
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, X):
        self.input = X
        self.pre_activation = X @ self.W + self.b
        self.output = get_activation(self.activation_name)(self.pre_activation)
        return self.output

    def backward(self, delta, weight_decay=0.0):
        # Added weight decay (L2 penalty) directly to the weight gradients
        self.grad_W = self.input.T @ delta + (weight_decay * self.W)
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        return delta @ self.W.T