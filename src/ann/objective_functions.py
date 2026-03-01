import numpy as np

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def mse(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

def mse_derivative(y_true, y_pred):
    return 2.0 * (y_pred - y_true) / y_true.shape[0]

LOSSES = {"cross_entropy": cross_entropy, "mean_squared_error": mse}
LOSS_DERIVATIVES = {"cross_entropy": cross_entropy_derivative, "mean_squared_error": mse_derivative}

def get_loss(name):
    return LOSSES[name]

def get_loss_derivative(name):
    return LOSS_DERIVATIVES[name]