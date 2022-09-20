from module import Module

class MSELoss(Module):
    def __init__(self):
        """ Mean Squared Error loss """
        super().__init__()

    def forward(self, y_pred, y_true):
        """ Forward pass """
        return ((y_pred - y_true) ** 2).mean()

    def __repr__(self):
        return f"MSE()"