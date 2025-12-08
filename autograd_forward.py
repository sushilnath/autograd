import numpy as np

class GradNode:
    def __init__(self, x, dx = 1.0):
        self.x = x
        self.dx = dx
    def _to_node(self, other):
        if isinstance(other, GradNode):
            return other
        return GradNode(other, 0.0)
    def __add__(self, other):
        other = self._to_node(other)
        return GradNode(self.x + other.x, self.dx + other.dx)
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        other = self._to_node(other)
        return GradNode(self.x - other.x, self.dx - other.dx)
    def __rsub__(self, other):
        other = self._to_node(other)
        return other.__sub__(self)
    def __mul__(self, other):
        other = self._to_node(other)
        return GradNode(self.x * other.x, self.x * other.dx + other.x * self.dx)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        other = self._to_node(other)
        return GradNode(self.x / other.x, (other.x * self.dx - self.x * other.dx) / (other.x ** 2))
    def __rtruediv__(self, other):
        other = self._to_node(other)
        return other.__truediv__(self)
    def __pow__(self, power):
        return GradNode(self.x ** power, power * (self.x ** (power - 1)) * self.dx)
    def __neg__(self):
        return GradNode(-self.x, -self.dx)
    def __repr__(self):
        return f"Node(x={self.x}, dx={self.dx})"

def sin(x):
    if isinstance(x, GradNode):
        return GradNode(np.sin(x.x), np.cos(x.x) * x.dx)
    return np.sin(x)

def cos(x):
    if isinstance(x, GradNode):
        return GradNode(np.cos(x.x), -np.sin(x.x) * x.dx)
    return np.cos(x)

def exp(x):
    if isinstance(x, GradNode):
        return GradNode(np.exp(x.x), np.exp(x.x) * x.dx)
    return np.exp(x)

def log(x):
    if isinstance(x, GradNode):
        return GradNode(np.log(x.x), (1 / x.x) * x.dx)
    return np.log(x)

