import math
from node import Node
from tensor import Tensor

import string
import random

class Sigmoid(Node):
    def __init__(self, name='', requires_grad=True):
        """ Initialize a node.
        :param name: The name of the node.
        :param requires_grad: Whether the node requires gradient.
        """
        self._name = name if name != '' else 'sigmoid+' + \
                            ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self._requires_grad = requires_grad

    def __repr__(self):
        """ Return a string representation of the node. """
        str_ = f'Sigmoid('
        str_ += f'name={self._name}, ' if self._name != '' else ''
        str_ += f'value={self.value}'
        str_ += f', grad={self._grad}' if self._requires_grad else ''
        str_ += f')'
        return f'{str_}'

    def __call__(self, x):
        """ Apply the sigmoid function to the input node. 
        :param x: The input node.
        :return: The output node.
        """
        x = x if 'Node' in str(type(x)) or 'Tensor' in str(type(x)) \
              else Node(x) if isinstance(x, (int, float)) else Tensor(x)

        if 'Tensor' in str(type(x)):
            if len(x.shape) == 2:
                out = Tensor([[Node(1.) / (Node(1.) +  (-x._data[i][j]).exp() ) \
                                for j in range(x.shape[1])] \
                                for i in range(x.shape[0])], requires_grad=self._requires_grad)

                self._value = out.items()
                if self._requires_grad:
                    def backward():
                        for i in range(x.shape[0]):
                            for j in range(x.shape[1]):
                                x._data[i][j]._grad += out._data[i][j]._value * (1 - out._data[i][j]._value) * out._data[i][j]._grad
                                x._data[i][j].backward()
                    out._backward = backward

            elif len(x.shape) == 3:
                out = Tensor([[[Node(1.) / (Node(1.) + (-x._data[i][j][k]).exp()) \
                                 for k in range(x.shape[2])] \
                                 for j in range(x.shape[1])] \
                                 for i in range(x.shape[0])], requires_grad=self._requires_grad)

                self._value = out.items()
                if self._requires_grad:
                    def backward():
                        for i in range(x.shape[0]):
                            for j in range(x.shape[1]):
                                for k in range(x.shape[2]):
                                    x._data[i][j][k]._grad += out._data[i][j][k]._value * (1 - out._data[i][j][k]._value) * out._data[i][j][k]._grad
                                    x._data[i][j][k].backward()
                    out._backward = backward
            else:
                raise ValueError(f'Invalid shape for Tensor: {x.shape}')

        else:
            out = Node(1 / (1 + math.exp(-x.value)), 
                       name=self._name,
                       requires_grad=self._requires_grad,
                       children_nodes=(x,), 
                       op='sigmoid')
            self._value = out.value
            if self._requires_grad:
                def backward():
                    x._grad += out.value * (1 - out.value) * out._grad
                out._backward = backward

        return out