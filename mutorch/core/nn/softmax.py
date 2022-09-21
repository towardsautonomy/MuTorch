import math
import core
from node import Node
from tensor import Tensor

import string
import random

class Softmax:
    """ Softmax function. """
    def __init__(self, name='', requires_grad=True):
        """ Initialize the node.
        :param name: The name of the node.
        :param requires_grad: Whether the node requires gradient.
        """
        self._name = name if name != '' else 'softmax+' + \
                            ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self._requires_grad = requires_grad

    def __repr__(self):
        """ Return a string representation of the node. """
        str_ = f'Softmax('
        str_ += f'name={self._name}, ' if self._name != '' else ''
        str_ += f'value={self.value}'
        str_ += f', grad={self._grad}' if self._requires_grad else ''
        str_ += f')'
        return f'{str_}'

    def __call__(self, x, normalize=False):
        """ Apply the softmax function to the input node. 
        :param x: The input node.
        :return: The output node.
        """
        x = x if 'Tensor' in str(type(x)) \
              else Tensor(x)

        if normalize:
            x = x - x.max()
        if sum(x.shape) <= 2:
            raise ValueError(f'Input tensor must have more than 1 element, but got {sum(x.shape)} elements.')
        elif len(x.shape) == 2:
            node_sum = Node(sum([math.exp(val) for val in x.items()]))
            out = Tensor([[(x._data[i][j].exp()) / node_sum \
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
            node_sum = Node(sum([math.exp(val) for val in x.items()]))
            out = Tensor([[[(x._data[i][j][k].exp()) / node_sum \
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
            raise ValueError(f'Input tensor must be 2D or 3D, but got {len(x.shape)}D tensor.')

        return out