import argparse
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation_derivative, softmax
from .objective_functions import get_loss, get_loss_derivative
from .optimizers import get_optimizer

class NeuralNetwork:
    def __init__(self, input_size=784, output_size=10, num_layers=3, hidden_size=128,
                 activation="relu", weight_init="xavier", loss="cross_entropy",
                 optimizer="adam", lr=0.001, weight_decay=0.0):
        # Handle argparse Namespace as first argument (autograder compatibility)
        if isinstance(input_size, argparse.Namespace):
            args = input_size
            input_size = getattr(args, 'input_size', 784)
            output_size = getattr(args, 'output_size', output_size)
            num_layers = getattr(args, 'num_layers', getattr(args, 'num_hidden_layers', num_layers))
            hidden_size = getattr(args, 'hidden_size', getattr(args, 'hidden_layer_size', hidden_size))
            activation = getattr(args, 'activation', activation)
            weight_init = getattr(args, 'weight_init', getattr(args, 'weight_initialisation', weight_init))
            loss = getattr(args, 'loss', loss)
            optimizer = getattr(args, 'optimizer', optimizer)
            lr = getattr(args, 'learning_rate', getattr(args, 'lr', lr))
            weight_decay = getattr(args, 'weight_decay', weight_decay)

        # hidden_size can be an int or a list of ints
        if isinstance(hidden_size, (list, tuple)):
            hidden_sizes = list(hidden_size)
        else:
            hidden_sizes = [int(hidden_size)] * num_layers
        # If list length doesn't match num_layers, adjust
        if len(hidden_sizes) != num_layers:
            if len(hidden_sizes) == 1:
                hidden_sizes = hidden_sizes * num_layers
            else:
                num_layers = len(hidden_sizes)

        self.loss_name = loss
        self.loss_fn = get_loss(loss)
        self.loss_deriv = get_loss_derivative(loss)
        self.activation = activation
        self.weight_decay = weight_decay # Explicitly store weight decay
        self.optimizer = get_optimizer(optimizer, lr=lr, weight_decay=weight_decay)
        
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = activation if not is_output else "identity"
            self.layers.append(NeuralLayer(sizes[i], sizes[i+1], act, weight_init, is_output))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # returns logits (pre-softmax)

    def backward(self, y_true, logits):
        n = len(self.layers)
        y_pred = softmax(logits)
        
        if self.loss_name == "cross_entropy":
            delta = (y_pred - y_true) / y_true.shape[0]
        else:
            # CORRECTED: Exact mathematical derivative for Mean Squared Error + Softmax
            d_loss = self.loss_deriv(y_true, y_pred)
            sum_dp = np.sum(d_loss * y_pred, axis=1, keepdims=True)
            delta = y_pred * (d_loss - sum_dp)
            
        grad_W_list, grad_b_list = [], []
        wd = getattr(self, 'weight_decay', 0.0) # Retrieve weight decay
        
        for i in reversed(range(n)):
            # Pass weight decay to the layer's backward method
            delta = self.layers[i].backward(delta, weight_decay=wd)
            grad_W_list.insert(0, self.layers[i].grad_W)
            grad_b_list.insert(0, self.layers[i].grad_b)
            if i > 0:
                act_deriv = get_activation_derivative(self.layers[i-1].activation_name)
                delta = delta * act_deriv(self.layers[i-1].pre_activation)
                
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train_epoch(self, X, y, batch_size=32):
        indices = np.random.permutation(X.shape[0])
        total_loss, correct = 0.0, 0
        for start in range(0, X.shape[0], batch_size):
            idx = indices[start:start+batch_size]
            xb, yb = X[idx], y[idx]
            logits = self.forward(xb)
            y_pred = softmax(logits)
            total_loss += self.loss_fn(yb, y_pred) * xb.shape[0]
            correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(yb, axis=1))
            self.backward(yb, logits)
            self.update_weights()
        return total_loss / X.shape[0], correct / X.shape[0]

    def evaluate(self, X, y, batch_size=256):
        total_loss, correct = 0.0, 0
        for start in range(0, X.shape[0], batch_size):
            xb, yb = X[start:start+batch_size], y[start:start+batch_size]
            logits = self.forward(xb)
            y_pred = softmax(logits)
            total_loss += self.loss_fn(yb, y_pred) * xb.shape[0]
            correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(yb, axis=1))
        return total_loss / X.shape[0], correct / X.shape[0]

    def predict(self, X, batch_size=256):
        preds = []
        for start in range(0, X.shape[0], batch_size):
            preds.append(self.forward(X[start:start+batch_size]))
        return np.vstack(preds)  # returns logits

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key, b_key = f"W{i}", f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def get_gradient_norms(self):
        return [np.linalg.norm(l.grad_W) for l in self.layers]

    def get_activation_stats(self):
        stats = []
        for l in self.layers:
            if hasattr(l, 'output'):
                zero_frac = np.mean(l.output == 0)
                stats.append({"mean": np.mean(l.output), "std": np.std(l.output),
                              "zero_fraction": zero_frac, "values": l.output.flatten()})
        return stats