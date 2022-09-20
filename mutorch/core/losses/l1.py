from module import Module

class L1Loss(Module):
    def __init__(self):
        """ Mean Squared Error loss """
        super().__init__()

    def forward(self, y_pred, y_true):
        """ Forward pass """
        return (y_pred - y_true).abs().mean()

    def __repr__(self):
        return f"L1()"

class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        """ Smooth L1 loss
        :param beta: beta parameter
        """
        super().__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        """ Forward pass """
        diff = (y_pred - y_true).abs()
        if diff < self.beta:
            return 0.5 * diff ** 2 / self.beta
        else:
            return diff - 0.5 * self.beta

    def __repr__(self):
        return f"SmoothL1(beta={self.beta})"