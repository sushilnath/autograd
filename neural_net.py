import random
import math
from autograd_backward import Node

class Neuron:
    def __init__(self, input_size):
        self.weights = [Node(random.uniform(-0.1, 0.1)) for _ in range(input_size)]
        self.bias = Node(0.1)

    def forward(self, inputs):
        output = Node(0.0)
        for x, w in zip(inputs, self.weights):
            output = output + x * w
        output = output + self.bias
        return output.relu()

class NeuralNet:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = [Neuron(layer_sizes[i]) for _ in range(layer_sizes[i + 1])]
            self.layers.append(layer)

    def flow(self, inputs):
        inputs = inputs.copy()
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                output = neuron.forward(inputs)
                outputs.append(output)
            inputs = outputs
        return inputs
    
    def train(self, inputs, targets, learning_rate=0.01):
        flow_outputs = self.flow(inputs)
        loss = sum((flow_outputs[i] + (- targets[i])) ** 2 for i in range(len(targets)))
        loss.backprop()
        for layer in self.layers:
            for neuron in layer:
                for i in range(len(neuron.weights)):
                    neuron.weights[i].x -= learning_rate * neuron.weights[i].grad
                    neuron.weights[i].grad = 0.0
                neuron.bias.x -= learning_rate * neuron.bias.grad
                neuron.bias.grad = 0.0
    
    def train_multiple_times(self, inputs, targets, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.train(inputs, targets, learning_rate)
    
    def predict(self, inputs):
        outputs = self.flow(inputs)
        return outputs

    def evaluate(self, inputs, targets):
        outputs = self.flow(inputs)
        loss = sum((outputs[i] + (- targets[i])) ** 2 for i in range(len(targets)))
        return loss.x
