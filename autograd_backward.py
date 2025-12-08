import numpy as np
class Node:
    def __init__(self, x, grad = 0.0, children=()):
        self.x = x          # Value of the node
        self.grad = grad    # Gradient of the node
        self.children = set(children)  # Child nodes
        self.propagate_fn = lambda: None  # Function to propagate gradients

    def _to_node(self, other):
        if isinstance(other, Node):
            return other
        # this means other is a constant
        return Node(other)

    def __add__(self, other):
        other = self._to_node(other)
        next_node = Node(self.x + other.x, 0.0, (self, other))
        def propagate():
            self.grad += next_node.grad
            other.grad += next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    def __radd__(self, other):
        other = self._to_node(other)
        return self.__add__(other)
    
    def __mul__(self, other):
        other = self._to_node(other)
        next_node = Node(self.x * other.x, 0.0, (self, other))
        def propagate():
            self.grad += other.x * next_node.grad
            other.grad += self.x * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self._to_node(other)
        next_node = Node(self.x / other.x, 0.0, (self, other))
        def propagate():
            self.grad += (1 / other.x) * next_node.grad
            other.grad += (-self.x / (other.x ** 2)) * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    def __rtruediv__(self, other):
        other = self._to_node(other)
        return other.__truediv__(self)
    
    def __pow__(self, power):
        next_node = Node(self.x ** power, 0.0, (self,))
        def propagate():
            self.grad += power * (self.x ** (power - 1)) * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    
    def __neg__(self):
        next_node = Node(-self.x, 0.0, (self,))
        def propagate():
            self.grad += -1 * next_node.grad
        next_node.propagate_fn = propagate
        return next_node

    def _rdag(self, visited, node_list):
        if self in visited:
            return
        visited.add(self)
        for child in self.children:
            child._rdag(visited, node_list)
        node_list.append(self)

    def relu(self):
        if self.x > 0:
            next_node = Node(self.x, 0.0, (self,))
            def propagate():
                self.grad += next_node.grad
            next_node.propagate_fn = propagate
            return next_node
        else:
            next_node = Node(0.0, 0.0, (self,))
            def propagate():
                self.grad += 0.0
            next_node.propagate_fn = propagate
            return next_node

    def backprop(self):
        self.grad = 1.0  # initialize the gradient
        rdag_list = []
        self._rdag(set(), rdag_list)
        for node in reversed(rdag_list):
            node.propagate_fn()

def sin(node):
    if isinstance(node, Node):
        next_node = Node(np.sin(node.x), 0.0, (node,))
        def propagate():
            node.grad += np.cos(node.x) * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    return np.sin(node)

def cos(node):
    if isinstance(node, Node):
        next_node = Node(np.cos(node.x), 0.0, (node,))
        def propagate():
            node.grad += -np.sin(node.x) * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    return np.cos(node)

def log(node):
    if isinstance(node, Node):
        next_node = Node(np.log(node.x), 0.0, (node,))
        def propagate():
            node.grad += (1 / node.x) * next_node.grad
        next_node.propagate_fn = propagate
        return next_node
    return np.log(node)