from autograd_backward import Node

class Neuron:
    def __init__(self, input_size):
        self.weights = Node(0.0) * input_size
        self.bias = Node(0.0)

    def forward(self, inputs):
        output = Node(0.0)
        for input in inputs:
            output += input * self.weights[inputs.index(input)]
        output += self.bias
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