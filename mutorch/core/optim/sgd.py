from module import Module

class SGD(Module):
    def __init__(self, parameters, lr=0.001):
        """ Stochastic Gradient Descent optimizer
        :param parameters: list of parameters to optimize
        :param lr: learning rate
        """
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """ Performs a single optimization step. """
        for param in self.parameters:
            param._value -= self.lr * param.grad

    def zero_grad(self):
        """ Sets gradients of all optimized parameters to zero. """
        for param in self.parameters:
            param.zero_grad()

    def __repr__(self):
        return f"SGD(lr={self.lr})"