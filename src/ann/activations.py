import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float64)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

ACTIVATIONS = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu, "softmax": softmax}
ACTIVATION_DERIVATIVES = {"sigmoid": sigmoid_derivative, "tanh": tanh_derivative, "relu": relu_derivative}

def get_activation(name):
    return ACTIVATIONS[name]

def get_activation_derivative(name):
    return ACTIVATION_DERIVATIVES[name]