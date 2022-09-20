from module import Module

class Sequential(Module):
    def __init__(self, *layers):
        """ Sequential model
        :param layers: list of layers
        """
        super().__init__()
        self.layers = layers
        # initialize children layers
        for i in range(1, len(self.layers)):
            self.layers[i]._children_layers += (self.layers[i-1],)

        self._parameters = [p for l in self.layers for p in l.parameters()]

    def forward(self, inputs):
        """ Forward pass 
        :param inputs: the inputs to the model
        :return: the output of the model
        """
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        """ Backward pass """
        for param in self._parameters:
            param.backward()

    def save(self, filename):
        """ Save the model to a file
        :param filename: the filename to save the model to
        """
        import pickle
        with open(filename, 'wb') as f:
            param_values = [p.value for p in self.parameters()]
            state_dict = {'parameters': param_values}
            pickle.dump(state_dict, f)

    def load(self, filename):
        """ Load the weights from a file
        :param filename: the filename to load the weights from
        """
        import pickle
        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)
            param_values = state_dict['parameters']
            for i, p in enumerate(self.parameters()):
                p._value = param_values[i]

    def draw_graph(self, filename='sequential_graph'):
        """ Draw the reversed graph of nodes in a top-down manner. """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        layer_labels = {}
        prev_layer = None
        for layer in self.layers:
            if prev_layer is None:
                G.add_node('input', shape='box')
                G.add_edge('input', layer)
            else:
                G.add_edge(prev_layer, layer)
            layer_labels[layer] = layer
            prev_layer = layer

        # add output node
        G.add_edge(prev_layer, 'out', shape='box')

        graph = nx.drawing.nx_pydot.to_pydot(G)
        graph.write_png(filename + '.png')

    def __repr__(self):
        layers_str = '\n\t'.join([str(l) for l in self.layers])
        return f"Sequential(\n\t{layers_str})"