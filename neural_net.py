import random
import math
from autograd_backward import Node

class Neuron:
    def __init__(self, input_size):
        self.weights = [Node(random.uniform(-0.5, 0.5)) for _ in range(input_size)]
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
    
def test_circle_visual_ascii(nn):
    print("\n--- Visualizing Decision Boundary (ASCII) ---")
    print("Legend: '.' = Inside Circle, '#' = Outside, ' ' = Unsure")
    
    # Grid range: y from 1.0 down to -1.0, x from -1.0 to 1.0
    steps = 20
    for y in range(steps, -steps - 1, -1):
        line = ""
        for x in range(-steps, steps + 1):
            val_x = x / steps
            val_y = y / steps
            
            # Predict
            pred = nn.predict([Node(val_x), Node(val_y)])[0].x
            
            # Character map based on prediction confidence
            if pred > 0.8:
                char = "."  # Confidently Inside
            elif pred < 0.2:
                char = "#"  # Confidently Outside
            else:
                char = " "  # Boundary / Unsure
            line += f"{char} "
        print(line)
    
def train_for_inside_circle(nn):
    training_data = []
    targets = []
    for _ in range(500):
        x = Node(random.uniform(-1, 1))
        y = Node(random.uniform(-1, 1))
        training_data.append([x, y])
        distance = math.sqrt(x.x ** 2 + y.x ** 2)
        targets.append([Node(1.0) if distance <= 0.5 else Node(0.0)])
    for epoch in range(1000):
        for i in range(len(training_data)):
            nn.train(training_data[i], targets[i], learning_rate=0.01)
    
    # Test Point 1: Inside Circle
    test_data_in = [Node(0.1), Node(0.1)]
    pred_in = nn.predict(test_data_in)[0].x
    
    # Test Point 2: Outside Circle
    test_data_out = [Node(0.9), Node(0.9)]
    pred_out = nn.predict(test_data_out)[0].x
    
    print("-" * 30)
    print(f"Point (0.1, 0.1) [Target 1.0]: {pred_in:.4f}")
    print(f"Point (0.9, 0.9) [Target 0.0]: {pred_out:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    nn = NeuralNet([2, 16, 1])
    train_for_inside_circle(nn)
    test_circle_visual_ascii(nn)
