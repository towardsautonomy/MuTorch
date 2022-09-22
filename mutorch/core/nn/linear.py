import random
from module import Module
from tensor import Tensor
from relu import ReLU

class Neuron(Module):
    def __init__(self, input_size,
                       weight_initializer=lambda: random.uniform(-1, 1),
                       bias_initializer=lambda: random.uniform(-1, 1)
                ):
        """  A single neuron
        :param input_size: the number of inputs
        :param weight_initializer: a function that returns a random weight
        :param bias_initializer: a function that returns a random bias
        """
        super().__init__()
        self.weights = Tensor([weight_initializer() for _ in range(input_size)], requires_grad=True)
        self.bias = Tensor(bias_initializer(), requires_grad=True)
        # internal parameters
        self._parameters = self.weights.data[0] + self.bias.data[0]

    def forward(self, inputs):
        """ Forward pass
        :param inputs: the inputs to the neuron
        :return: the output of the neuron
        """
        inputs = Tensor(inputs) if not 'Tensor' in str(type(inputs)) else inputs
        out = (inputs * self.weights).sum() + self.bias
        # get the output value node
        return out.data[0][0] if 'Tensor' in str(type(out)) else out

    def __repr__(self):
        return f"Neuron(input_size={len(self.weights)})"

class Linear(Module):
    def __init__(self, input_size, 
                       output_size,
                       weight_initializer=lambda: random.uniform(-1, 1),
                       bias_initializer=lambda: random.uniform(-1, 1),
                       activation=ReLU(),
                       children_layers=()):
        """ A linear layer
        :param input_size: the number of inputs
        :param output_size: the number of outputs
        :param weight_initializer: a function that returns a random weight
        :param bias_initializer: a function that returns a random bias
        :param activation: the activation function
        :param children_layers: the children layers
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.neurons = [Neuron(input_size, 
                               weight_initializer, 
                               bias_initializer) for _ in range(output_size)]
        # internal parameters
        self._parameters = [p for n in self.neurons for p in n.parameters()]
        self._children_layers = children_layers

    def forward(self, inputs):
        """ A forward pass through the layer
        :param inputs: the inputs to the linear layer
        :return: the output of the linear layer
        """
        inputs = Tensor(inputs) if not 'Tensor' in str(type(inputs)) else inputs
        batch_size = inputs.shape[0]
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"Input size must be {self.input_size} but got {inputs.shape[1]}")
        if batch_size != 1:
            out = Tensor([[n.forward(inputs.data[i]) for n in self.neurons] for i in range(batch_size)])
        else:
            out = Tensor([n.forward(inputs) for n in self.neurons])
        out = self.activation(out) if self.activation else out

        return out

    def __repr__(self):
        layer_str = f"Linear(input_size={self.input_size}, output_size={self.output_size}"
        layer_str += f", activation={self.activation.name}" if self.activation else ""
        layer_str += ")"
        return layer_str