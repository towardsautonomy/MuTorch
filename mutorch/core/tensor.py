import core
from node import Node

class Tensor:
    def __init__(self, data, requires_grad=True):
        """ Initialize a tensor.
        
        Args:
            data: The data of the tensor.
            requires_grad: Whether the tensor requires gradient.
        """
        self._data = data
        self._shape = None
        self._backward = lambda: None
        self.requires_grad = requires_grad

        ## check if data is a scalar or a list of scalars
        if isinstance(data, (int, float)) or \
           isinstance(data, list) and isinstance(data[0], (int, float)) or \
           isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], (int, float)) or \
           isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], list) and isinstance(data[0][0][0], (int, float)):
            self._convert_to_tensor(data, requires_grad, convert_to_node=True)

        ## check if data is a Node or a list of Nodes
        elif isinstance(data, Node) or \
             isinstance(data, list) and isinstance(data[0], Node) or \
             isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], Node) or \
             isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], list) and isinstance(data[0][0][0], Node):
            self._convert_to_tensor(data, requires_grad, convert_to_node=False)

        ## check if data is a list of tensors
        elif isinstance(data, list) and isinstance(data[0], (Tensor, core.tensor.Tensor)) or \
                isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], (Tensor, core.tensor.Tensor)) or \
                isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], list) and isinstance(data[0][0][0], (Tensor, core.tensor.Tensor)):
            self._shape = (len(data), len(data[0]._data))
            self._data = [row._data for row in data]
        else:
            raise ValueError('Data must be scalar/Node or a list of scalars/Nodes.')

    def _convert_to_tensor(self, data, requires_grad, convert_to_node):
        """ Convert a scalar or a list of scalars to a tensor. """
        # check if data is a scalar
        if isinstance(data, (int, float, Node)):
            self._shape = (1,1)
            self._data = [[Node(data, requires_grad=requires_grad) \
                            if convert_to_node else data]]
        # check if data is a list
        elif isinstance(data, list) and not isinstance(data[0], list):
            self._shape = (1, len(data))
            self._data = [[Node(x, requires_grad=requires_grad) \
                            if convert_to_node else x for x in data]]
            self._backward = lambda: [node.backward() for row in self._data for node in row]
        # check if data is a 2D list
        elif isinstance(data, list) and \
             isinstance(data[0], list) and \
             not isinstance(data[0][0], list):
            self._shape = (len(data), len(data[0]))
            self._data = [[Node(x, requires_grad=requires_grad) \
                            if convert_to_node else x for x in row] \
                            for row in data]
            self._backward = lambda: [node.backward() for row in self._data for node in row]
        # check if data is a 3D list
        elif isinstance(data, list) and \
             isinstance(data[0], list) and \
             isinstance(data[0][0], list) and \
             not isinstance(data[0][0][0], list):
            self._shape = (len(data), len(data[0]), len(data[0][0]))
            self._data = [[[Node(x, requires_grad=requires_grad) \
                            if convert_to_node else x for x in row] \
                            for row in data] \
                            for data in data]
            self._backward = lambda: [[node.backward() for row in data for node in row] for data in self._data]
        # check if data is a 4D list
        elif isinstance(data, list) and \
             isinstance(data[0], list) and \
             isinstance(data[0][0], list) and \
             isinstance(data[0][0][0], list) and \
             not isinstance(data[0][0][0][0], list):
            self._shape = (len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))
            self._data = [[[[Node(x, requires_grad=requires_grad) \
                            if convert_to_node else x for x in row] \
                            for row in data] \
                            for data in data] \
                            for data in data]
            self._backward = lambda: [[[node.backward() for row in data for node in row] for data in data] for data in self._data]

        # we don't support tensors with more than 4 dimensions
        else:
            raise ValueError('The tensor must have a maximum of 4 dimensions.')
        
    @property
    def shape(self):
        """ Return the shape of the tensor. """
        return self._shape

    @property
    def data(self):
        """ Return the data of the tensor. """
        return self._data

    def __repr__(self):
        """ Return a string representation of the tensor. """
        str_ = f'Tensor('
        str_ += f'shape={self._shape}, \n\t'
        data_str = '\n\t'.join([str(x) for x in self._data])
        str_ += f'data=[{data_str}'
        str_ += ')]'
        str_ += f', requires_grad={self.requires_grad}'
        str_ += f')'
        return f'{str_}'

    def __add__(self, other):
        """ Add a tensor to another tensor or a scalar. """
        other = other if isinstance(other, (Tensor, core.tensor.Tensor)) else Tensor(other)
        assert self.shape == other.shape or \
               other.shape == (1,self.shape[0]) or \
               other.shape == (1,1), \
                f'The shapes of the tensors must be the same, or one tensor should have the shape: (1, {other.shape[0]}) or (1,). self.shape = {self.shape}, other.shape = {other.shape}'
        
        if other.shape == (1,1):
            # broadcast the scalar to the shape of the tensor
            other = Tensor([[other.data[0][0] for _ in range(self.shape[1])] \
                            for _ in range(self.shape[0])])

        if other.shape == (1,self.shape[0]):
            out = Tensor([[self._data[i][j] + other._data[i][0] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        else:
            out = Tensor([[self._data[i][j] + other._data[i][j] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        return out

    def __radd__(self, other):
        """ Add a tensor to another tensor or a scalar. """
        return self.__add__(other)

    def __sub__(self, other):
        """ Subtract a tensor from another tensor or a scalar. """
        other = other if isinstance(other, (Tensor, core.tensor.Tensor)) else Tensor(other)
        assert self.shape == other.shape or \
               other.shape == (1,self.shape[0]) or \
               other.shape == (1,1), \
                f'The shapes of the tensors must be the same, or one tensor should have the shape: (1, {other.shape[0]}) or (1,1). self.shape = {self.shape}, other.shape = {other.shape}'
        if other.shape == (1,1):
            # broadcast the scalar to the shape of the tensor
            other = Tensor([[other.data[0][0] for _ in range(self.shape[1])] \
                            for _ in range(self.shape[0])])
        if other.shape == (1,self.shape[0]):
            out = Tensor([[self._data[i][j] - other._data[i][0] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        else:
            out = Tensor([[self._data[i][j] - other._data[i][j] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)   
        return out

    def __rsub__(self, other):
        """ Subtract a tensor from another tensor or a scalar. """
        return self.__sub__(other)

    def __mul__(self, other):
        """ Multiply a tensor with another tensor or a scalar. """
        other = other if isinstance(other, (Tensor, core.tensor.Tensor)) else Tensor(other)
        assert self.shape == other.shape or \
               other.shape == (1,self.shape[0]) or \
               other.shape == (1,1), \
                f'The shapes of the tensors must be the same, or one tensor should have the shape: (1, {other.shape[0]}) or (1,1). self.shape = {self.shape}, other.shape = {other.shape}'

        if other.shape == (1,1):
            # broadcast the scalar to the shape of the tensor
            other = Tensor([[other.data[0][0] for _ in range(self.shape[1])] \
                            for _ in range(self.shape[0])])

        if other.shape == (1,self.shape[0]):
            out = Tensor([[self._data[i][j] * other._data[i][0] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        else:
            out = Tensor([[self._data[i][j] * other._data[i][j] \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        return out

    def __rmul__(self, other):
        """ Multiply a tensor with another tensor or a scalar. """
        return self.__mul__(other)

    def __pow__(self, other):
        """ Raise a tensor to the power of another tensor or a scalar. """
        other = other if isinstance(other, (Tensor, core.tensor.Tensor)) else Tensor(other)
        assert other.shape == (1,self.shape[0]) or \
               other.shape == (1,1), \
                f'The shapes of the tensors must be (1, {other.shape[0]}) or (1,1).'
        if other.shape == (1,1):
            # broadcast the scalar to the shape of the tensor
            other = Tensor([[other.data[0][0] for _ in range(self.shape[1])] \
                            for _ in range(self.shape[0])])
        out = Tensor([[self._data[i][j] ** other._data[i][0] \
                        for j in range(self.shape[1])] \
                        for i in range(self.shape[0])], \
                        requires_grad=self.requires_grad)
        return out

    def __rpow__(self, other):
        """ Raise a tensor to the power of another tensor or a scalar. """
        return self.__pow__(other)

    def __truediv__(self, other):
        """ Divide a tensor by another tensor or a scalar. """
        other = other if isinstance(other, (Tensor, core.tensor.Tensor)) else Tensor(other)
        assert self.shape == other.shape or \
                other.shape == (1,self.shape[0]) or \
               other.shape == (1,1), \
                f'The shapes of the tensors must be the same, or one tensor should have the shape: (1, {other.shape[0]}) or (1,1). self.shape = {self.shape}, other.shape = {other.shape}'
        if other.shape == (1,1):
            # broadcast the scalar to the shape of the tensor
            other = Tensor([[other.data[0][0] for _ in range(self.shape[1])] \
                            for _ in range(self.shape[0])])
        if other.shape == (1,self.shape[0]):
            out = Tensor([[self._data[i][j] * ( other._data[i][0] ** -1 ) \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        else:
            out = Tensor([[self._data[i][j] * ( other._data[i][j] ** -1 ) \
                            for j in range(self.shape[1])] \
                            for i in range(self.shape[0])], \
                            requires_grad=self.requires_grad)
        return out

    def __rtruediv__(self, other):  
        """ Divide a tensor by another tensor or a scalar. """
        return self.__truediv__(other)

    def __neg__(self):
        """ Negate a tensor. """
        out = Tensor([[-self._data[i][j] \
                        for j in range(self.shape[1])] \
                        for i in range(self.shape[0])], \
                        requires_grad=self.requires_grad)
        return out

    def exp(self, e):
        """ Calculate the exponential of a tensor. """
        out = Tensor([[self._data[i][j] ** e \
                        for j in range(self.shape[1])] \
                        for i in range(self.shape[0])], \
                        requires_grad=self.requires_grad)
        return out

    def sum(self):
        """ Sum the elements of the tensor. """
        sum_ = Node(0, requires_grad=self.requires_grad)
        for row in self._data:
            for x in row:
                sum_ += x
        return Tensor(sum_)

    def mean(self):
        """ Compute the mean of the elements of the tensor. """
        return self.sum() / (self.shape[0] * self.shape[1])

    def item(self):
        """ Return the value of the tensor as a python number. """
        assert self.shape == (1,1), f'The shape of the tensor must be (1,1). self.shape = {self.shape}'
        return self._data[0][0]._value

    def items(self):
        """ Return the value of the tensor as a python list. """
        flattened_tensor = self.flatten()
        return [flattened_tensor._data[0][i]._value for i in range(flattened_tensor.shape[1])]

    def flatten(self):
        """ Flatten the tensor. """
        if len(self.shape) == 2:
            return Tensor([self._data[i][j] \
                            for j in range(self.shape[1]) \
                            for i in range(self.shape[0])])
        elif len(self.shape) == 3:
            return Tensor([self._data[i][j][k] \
                            for k in range(self.shape[2]) \
                            for j in range(self.shape[1]) \
                            for i in range(self.shape[0])])
        elif len(self.shape) == 4:
            return Tensor([self._data[i][j][k][l] \
                            for l in range(self.shape[3]) \
                            for k in range(self.shape[2]) \
                            for j in range(self.shape[1]) \
                            for i in range(self.shape[0])])
        else:
            raise ValueError(f'The shape of the tensor is not supported. self.shape = {self.shape}')

    def backward(self):
        """ Backpropagate the gradient through the computational graph. """
        if self.requires_grad:
            self._backward()