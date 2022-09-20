import math
import core
from node import Node
from tensor import Tensor

class Tanh(Node):
    def __init__(self, name='tanh', requires_grad=True):
        """ Initialize a node.
        :param name: The name of the node.
        :param requires_grad: Whether the node requires gradient.
        """
        self._value = 0.
        self._name = name
        self._requires_grad = requires_grad

    def __repr__(self):
        """ Return a string representation of the node. """
        str_ = f'Tanh('
        str_ += f'name={self.name}, ' if self.name != '' else ''
        str_ += f'value={self.value}'
        str_ += f', grad={self.grad}' if self._requires_grad else ''
        str_ += f')'
        return f'{str_}'

    def __call__(self, x):
        """ Apply the tanh function to the input node. 
        :param x: The input node.
        :return: The output node.
        """
        x = x if isinstance(x, (Node, core.node.Node, Tensor, core.tensor.Tensor)) \
              else Node(x) if isinstance(x, (int, float)) else Tensor(x)

        if isinstance(x, (Tensor, core.tensor.Tensor)):
            if len(x.shape) == 2:
                out = Tensor([[x._data[i][j].tanh()  \
                                for j in range(x.shape[1])] \
                                for i in range(x.shape[0])], requires_grad=self._requires_grad)

                self._value = out.items()
                if self._requires_grad:
                    def backward():
                        for i in range(x.shape[0]):
                            for j in range(x.shape[1]):
                                x._data[i][j]._grad += (1 - out._data[i][j]._value ** 2) * out._data[i][j]._grad
                                x._data[i][j].backward()
                    out._backward = backward

            elif len(x.shape) == 3:
                out = Tensor([[[x._data[i][j][k].tanh() \
                                 for k in range(x.shape[2])] \
                                 for j in range(x.shape[1])] \
                                 for i in range(x.shape[0])], requires_grad=self._requires_grad)

                self._value = out.items()
                if self._requires_grad:
                    def backward():
                        for i in range(x.shape[0]):
                            for j in range(x.shape[1]):
                                for k in range(x.shape[2]):
                                    x._data[i][j][k]._grad += (1 - out._data[i][j][k]._value ** 2) * out._data[i][j][k]._grad
                                    x._data[i][j][k].backward()
                    out._backward = backward

        else:
            out = Node(math.tanh(x.value), 
                       name=self._name,
                       requires_grad=self._requires_grad,
                       children_nodes=(x,), 
                       op='tanh')
            self._value = out.value
            if self._requires_grad:
                def backward():
                    x._grad += out._grad * (1 - out.value ** 2)
                out._backward = backward

        return out