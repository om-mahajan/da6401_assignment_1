import numpy as np

class SGD:
    def __init__(self, lr=0.001, weight_decay=0.0):
        self.lr, self.wd = lr, weight_decay
    def step(self, layers):
        for l in layers:
            l.grad_W += self.wd * l.W
            l.W -= self.lr * l.grad_W
            l.b -= self.lr * l.grad_b

class Momentum:
    def __init__(self, lr=0.001, beta=0.9, weight_decay=0.0):
        self.lr, self.beta, self.wd = lr, beta, weight_decay
        self.v_w, self.v_b = {}, {}
    def step(self, layers):
        for i, l in enumerate(layers):
            if i not in self.v_w:
                self.v_w[i] = np.zeros_like(l.W)
                self.v_b[i] = np.zeros_like(l.b)
            l.grad_W += self.wd * l.W
            self.v_w[i] = self.beta * self.v_w[i] + l.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + l.grad_b
            l.W -= self.lr * self.v_w[i]
            l.b -= self.lr * self.v_b[i]

class NAG:
    def __init__(self, lr=0.001, beta=0.9, weight_decay=0.0):
        self.lr, self.beta, self.wd = lr, beta, weight_decay
        self.v_w, self.v_b = {}, {}
    def step(self, layers):
        for i, l in enumerate(layers):
            if i not in self.v_w:
                self.v_w[i] = np.zeros_like(l.W)
                self.v_b[i] = np.zeros_like(l.b)
            l.grad_W += self.wd * l.W
            self.v_w[i] = self.beta * self.v_w[i] + l.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + l.grad_b
            l.W -= self.lr * (self.beta * self.v_w[i] + l.grad_W)
            l.b -= self.lr * (self.beta * self.v_b[i] + l.grad_b)

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr, self.beta, self.eps, self.wd = lr, beta, eps, weight_decay
        self.s_w, self.s_b = {}, {}
    def step(self, layers):
        for i, l in enumerate(layers):
            if i not in self.s_w:
                self.s_w[i] = np.zeros_like(l.W)
                self.s_b[i] = np.zeros_like(l.b)
            l.grad_W += self.wd * l.W
            self.s_w[i] = self.beta * self.s_w[i] + (1 - self.beta) * l.grad_W ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * l.grad_b ** 2
            l.W -= self.lr * l.grad_W / (np.sqrt(self.s_w[i]) + self.eps)
            l.b -= self.lr * l.grad_b / (np.sqrt(self.s_b[i]) + self.eps)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m_w, self.m_b, self.v_w, self.v_b = {}, {}, {}, {}
        self.t = 0
    def step(self, layers):
        self.t += 1
        for i, l in enumerate(layers):
            if i not in self.m_w:
                self.m_w[i] = np.zeros_like(l.W); self.v_w[i] = np.zeros_like(l.W)
                self.m_b[i] = np.zeros_like(l.b); self.v_b[i] = np.zeros_like(l.b)
            l.grad_W += self.wd * l.W
            self.m_w[i] = self.b1 * self.m_w[i] + (1 - self.b1) * l.grad_W
            self.v_w[i] = self.b2 * self.v_w[i] + (1 - self.b2) * l.grad_W ** 2
            self.m_b[i] = self.b1 * self.m_b[i] + (1 - self.b1) * l.grad_b
            self.v_b[i] = self.b2 * self.v_b[i] + (1 - self.b2) * l.grad_b ** 2
            mw_hat = self.m_w[i] / (1 - self.b1 ** self.t)
            vw_hat = self.v_w[i] / (1 - self.b2 ** self.t)
            mb_hat = self.m_b[i] / (1 - self.b1 ** self.t)
            vb_hat = self.v_b[i] / (1 - self.b2 ** self.t)
            l.W -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            l.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

class Nadam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m_w, self.m_b, self.v_w, self.v_b = {}, {}, {}, {}
        self.t = 0
    def step(self, layers):
        self.t += 1
        for i, l in enumerate(layers):
            if i not in self.m_w:
                self.m_w[i] = np.zeros_like(l.W); self.v_w[i] = np.zeros_like(l.W)
                self.m_b[i] = np.zeros_like(l.b); self.v_b[i] = np.zeros_like(l.b)
            l.grad_W += self.wd * l.W
            self.m_w[i] = self.b1 * self.m_w[i] + (1 - self.b1) * l.grad_W
            self.v_w[i] = self.b2 * self.v_w[i] + (1 - self.b2) * l.grad_W ** 2
            self.m_b[i] = self.b1 * self.m_b[i] + (1 - self.b1) * l.grad_b
            self.v_b[i] = self.b2 * self.v_b[i] + (1 - self.b2) * l.grad_b ** 2
            mw_hat = self.m_w[i] / (1 - self.b1 ** self.t)
            vw_hat = self.v_w[i] / (1 - self.b2 ** self.t)
            mb_hat = self.m_b[i] / (1 - self.b1 ** self.t)
            vb_hat = self.v_b[i] / (1 - self.b2 ** self.t)
            mw_nesterov = self.b1 * mw_hat + (1 - self.b1) * l.grad_W / (1 - self.b1 ** self.t)
            mb_nesterov = self.b1 * mb_hat + (1 - self.b1) * l.grad_b / (1 - self.b1 ** self.t)
            l.W -= self.lr * mw_nesterov / (np.sqrt(vw_hat) + self.eps)
            l.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "nag": NAG, "rmsprop": RMSProp, "adam": Adam, "nadam": Nadam}

def get_optimizer(name, lr=0.001, weight_decay=0.0, **kwargs):
    return OPTIMIZERS[name](lr=lr, weight_decay=weight_decay, **kwargs)