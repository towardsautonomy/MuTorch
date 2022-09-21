from module import Module

class NLLLoss(Module):
    def __init__(self):
        """ Negative Log Likelihood loss """
        super().__init__()

    def forward(self, y_pred, y_true):
        """ Forward pass """
        return -((y_true * y_pred.log()).mean())

    def __repr__(self):
        return f"NLL()"