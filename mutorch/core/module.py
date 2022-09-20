""" This file contains the definition of Module class. """

class Module:
    def __init__(self):
        """ Base class for all modules. """
        self._parameters = []

    def zero_grad(self):
        """ Sets gradients of all parameters to zero. """
        for param in self._parameters:
            param.zero_grad()

    def parameters(self):
        """ Returns a list of parameters. """
        return self._parameters

    def forward(self, *args, **kwargs):
        """ Forward pass. """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """ Enables the module to be called like a function. """
        return self.forward(*args, **kwargs)

    def num_parameters(self):
        """ Returns the number of parameters. """
        return len(self._parameters)
