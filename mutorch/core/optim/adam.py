import math
from module import Module

class Adam(Module):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """ Adam optimizer
        :param parameters: list of parameters to optimize
        :param lr: learning rate
        :param beta1: exponential decay rate for the first moment estimates
        :param beta2: exponential decay rate for the second-moment estimates
        :param eps: term added to the denominator to improve numerical stability
        """
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0] * len(parameters)
        self.v = [0] * len(parameters)

    def step(self):
        """ Performs a single optimization step. """
        self.t += 1
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param._value -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """ Sets gradients of all optimized parameters to zero. """
        for param in self.parameters:
            param.zero_grad()

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"