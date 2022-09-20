import math
import pprint

class Node:
    # default values
    _value = 0.
    _name = ''
    _grad = 0.

    def __init__(self, value, 
                       name='', 
                       requires_grad=True,
                       children_nodes=(),
                       op=None):
        """ Initialize a node. 
        :param value: the value of the node
        :param name: the name of the node
        :param requires_grad: whether the node requires gradient
        :param children_nodes: the children nodes
        :param op: the operation performed on the node
        """

        # make sure that the value is a scalar
        if not isinstance(value, (int, float)):
            raise TypeError('The value of the node must be a scalar.')

        # internal states
        self._value = float(value)
        self._name = name
        self._requires_grad = requires_grad
        self._children_nodes = children_nodes
        self._op = op
        self._grad = 0. if requires_grad else None
        self._backward = lambda: None

    def __repr__(self):
        """ Return a string representation of the node. """
        str_ = f'Node('
        str_ += f'name={self.name}, ' if self.name != '' else ''
        str_ += f'op={self._op}, ' if self._op is not None else ''
        str_ += f'value={self.value}'
        str_ += f', grad={self._grad}' if self.requires_grad else ''
        str_ += f')'
        return f'{str_}'

    def __add__(self, other):
        """ Add two nodes. 
        :param other: the other node
        :return: the sum of the two nodes
        """
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, 
                   children_nodes=(self, other), 
                   op='+')
        
        def backward():
            self._grad += out._grad
            other._grad += out._grad
        out._backward = backward

        return out

    def __radd__(self, other):
        """ Add two nodes. 
        :param other: the other node
        :return: the sum of the two nodes
        """
        return self.__add__(other)

    def __sub__(self, other):
        """ Subtract two nodes. 
        :param other: the other node
        :return: the difference of the two nodes
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """ Subtract two nodes. 
        :param other: the other node
        :return: the difference of the two nodes
        """
        return self.__sub__(other)

    def __mul__(self, other):
        """ Multiply two nodes. 
        :param other: the other node
        :return: the product of the two nodes
        """
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, 
                   children_nodes=(self, other), 
                   op='*')
        
        def backward():
            self._grad += other.value * out._grad
            other._grad += self.value * out._grad
        out._backward = backward

        return out

    def __rmul__(self, other):
        """ Multiply two nodes. 
        :param other: the other node
        :return: the product of the two nodes
        """
        return self.__mul__(other)

    def __pow__(self, other):
        """ Raise a node to a power. 
        :param other: the power
        :return: the node raised to the power
        """
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value ** other.value, 
                   children_nodes=(self, other), 
                   op='**')
        def backward():
            eps = 1e-8
            self._grad += other.value * self.value ** (other.value - 1) * out._grad
            # other._grad += self.value ** other.value * math.log(self.value + eps) * out._grad
        out._backward = backward

        return out

    def __rpow__(self, other):
        """ Raise a node to a power. 
        :param other: the power
        :return: the node raised to the power
        """
        return self.__pow__(other)

    def __truediv__(self, other):
        """ Divide two nodes. 
        :param other: the other node
        :return: the quotient of the two nodes
        """
        return self.__mul__(other ** -1)

    def __rtruediv__(self, other):
        """ Divide two nodes. 
        :param other: the other node
        :return: the quotient of the two nodes
        """
        return self.__truediv__(other)

    def __neg__(self):
        """ Negate a node. 
        :return: the negated node
        """
        return self.__mul__(-1)

    def __abs__(self):
        """ Return the absolute value of a node. 
        :return: the absolute value of the node
        """
        return self.__mul__(self.value / abs(self.value))

    def tanh(self):
        """ Compute the hyperbolic tangent of a node. 
        :return: the hyperbolic tangent of the node
        """
        out = Node(math.tanh(self.value), 
                   children_nodes=(self,), 
                   op='tanh')
        
        def backward():
            self._grad += (1 - math.tanh(self.value) ** 2) * out._grad
        out._backward = backward

        return out

    def exp(self):
        """ Compute the exponential of a node. 
        :return: the exponential of the node
        """
        out = Node(math.exp(self.value), 
                   children_nodes=(self,), 
                   op='exp')
        
        def backward():
            self._grad += math.exp(self.value) * out._grad
        out._backward = backward

        return out

    def _build_node_graph(self):
        """ Build a graph of nodes. 
        :return: the graph of nodes
        """
        graph = {self: []}
        stack = [self]
        while stack:
            node = stack.pop()
            for child_node in node._children_nodes:
                graph[node] = graph.get(node, []) + [child_node]
                stack.append(child_node)

        return graph

    def backward(self):
        """ Backpropagate the gradient. 
        """
        self._grad = 1.

        # build a stack of nodes and perform backward propagation
        stack = [self]
        while stack:
            node = stack.pop()
            node._backward()
            stack.extend(node._children_nodes)

    def zero_grad(self):
        """ Reset the gradient to zero. """
        self._grad = 0.

        # build a stack of nodes and set the gradient to zero
        stack = [self]
        while stack:
            node = stack.pop()
            node._grad = 0.
            stack.extend(node._children_nodes)

    @property
    def value(self):
        """ Return the value of the node. """
        return self._value

    @property
    def name(self):
        """ Return the name of the node. """
        return self._name

    @property
    def requires_grad(self):
        """ Return whether the node requires gradient. """
        return self._requires_grad

    @property
    def grad(self):
        """ Return the gradient of the node. """
        return self._grad

    def print_graph(self):
        """ Print the graph of nodes """
        pprint.pprint(self._build_node_graph())

    def draw_graph(self, show_grad=True, 
                         filename='node_graph'):
        """ Draw the reversed graph of nodes in a top-down manner. 
        :param show_grad: whether to show the gradient
        :param filename: the filename of the graph
        """
        import networkx as nx
        import pydot

        # plt.figure(figsize=(10, 10))
        graph = self._build_node_graph()
        G = nx.DiGraph()
        node_labels = {}
        for node, children_nodes in graph.items():
            # add nodes
            G.add_node(node, op=node._op)
            node_labels[node] = node.name if node.name != '' else node._op
            
            # add edges
            for child_node in children_nodes:
                edge_label = {'val' : float("%.2f" % child_node.value)}
                if child_node.requires_grad and show_grad:
                    edge_label['grad'] = float("%.2f" % child_node._grad)

                G.add_edge(child_node, node, **edge_label)
        
        # add output node
        G.add_node('out', op='out')
        G.add_edge(self, 'out', val=float("%.2f" % self.value))
        
        graph = nx.drawing.nx_pydot.to_pydot(G)
        # # convert dot file to png using pydot with squared nodes
        # (graph,) = pydot.graph_from_dot_file(filename + '.dot')
        graph.write_png(filename + '.png')