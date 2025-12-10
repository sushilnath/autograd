import math
import random
from neural_net import Neuron, NeuralNet
from autograd_backward import Node
from neural_net import Neuron, NeuralNet

def test_neuron_forward():
    neuron = Neuron(3)

    # Set weights and bias manually for testing
    neuron.weights = [Node(0.5), Node(-1.0), Node(2.0)]
    neuron.bias = Node(0.1)

    # Define inputs
    inputs = [Node(1.0), Node(2.0), Node(3.0)]

    # Perform forward pass
    output = neuron.forward(inputs)

    # Expected output calculation:
    # z = (1.0 * 0.5) + (2.0 * -1.0) + (3.0 * 2.0) + 0.1
    # z = 0.5 - 2.0 + 6.0 + 0.1 = 4.6
    expected_value = 4.6 # if 4.6 > 0 else 0.0  # relu activation

    assert abs(output.x - expected_value) < 1e-6, f"Neuron forward failed: Got {output.x}, Expected {expected_value}"
    print("Neuron forward test passed")

def test_neuron_forward_relu_negative():

    # Create a neuron with 2 inputs
    neuron = Neuron(2)

    # Set weights and bias manually for testing
    neuron.weights = [Node(-1.0), Node(-1.0)]
    neuron.bias = Node(-0.5)

    # Define inputs
    inputs = [Node(1.0), Node(1.0)]

    # Perform forward pass
    output = neuron.forward(inputs)

    # Expected output calculation:
    # z = (1.0 * -1.0) + (1.0 * -1.0) + (-0.5)
    # z = -1.0 - 1.0 - 0.5 = -2.5
    expected_value = 0.0  # relu activation

    assert abs(output.x - expected_value) < 1e-6, f"Neuron forward ReLU negative failed: Got {output.x}, Expected {expected_value}"
    print("Neuron forward ReLU negative test passed")

def test_neural_net_flow():
    # Create a simple neural network with 2 inputs, 1 hidden layer with 2 neurons, and 1 output neuron
    nn = NeuralNet([2, 2, 1])

    # Manually set weights and biases for testing
    # Hidden layer
    nn.layers[0][0].weights = [Node(0.5), Node(-1.0)]
    nn.layers[0][0].bias = Node(0.0)
    nn.layers[0][1].weights = [Node(1.0), Node(1.0)]
    nn.layers[0][1].bias = Node(0.0)
    # Output layer
    nn.layers[1][0].weights = [Node(1.0), Node(-1.0)]
    nn.layers[1][0].bias = Node(0.0)

    # Define inputs
    inputs = [Node(1.0), Node(2.0)]

    # Perform forward pass
    outputs = nn.flow(inputs)

    # Expected output calculation:
    # Hidden layer:
    # Neuron 1: z1 = (1.0 * 0.5) + (2.0 * -1.0) + 0.0 = 0.5 - 2.0 = -1.5 -> ReLU( -1.5 ) = 0.0
    # Neuron 2: z2 = (1.0 * 1.0) + (2.0 * 1.0) + 0.0 = 1.0 + 2.0 = 3.0 -> ReLU( 3.0 ) = 3.0
    # Output layer:
    # Neuron: z_out = (0.0 * 1.0) + (3.0 * -1.0) + 0.0 = 0.0 - 3.0 = -3.0 -> ReLU( -3.0 ) = 0.0
    expected_output = 0.0

    assert abs(outputs[0].x - expected_output) < 1e-6, f"NeuralNet flow failed: Got {outputs[0].x}, Expected {expected_output}"
    print("NeuralNet flow test passed")

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
    
def train_for_inside_circle(nn, iteraton_count = 100, check_progress=False):
    training_data = []
    targets = []
    for _ in range(500):
        x = Node(random.uniform(-1, 1))
        y = Node(random.uniform(-1, 1))
        training_data.append([x, y])
        distance = math.sqrt(x.x ** 2 + y.x ** 2)
        targets.append([Node(1.0) if distance <= 0.5 else Node(0.0)])
    for idx in range(iteraton_count):
        for i in range(len(training_data)):
            nn.train(training_data[i], targets[i], learning_rate=0.01)

        print("yoyo1")

        if (check_progress == False):
            continue
        
        if (idx % 5 == 0):
            print("yoyo2") 
            total_loss = 0.0
            for i in range(len(training_data)):
                inputs_copy = [Node(n.x) for n in training_data[i]]
                targets_copy = [Node(n.x) for n in targets[i]]
                flow_outputs = nn.flow(inputs_copy)
                loss = sum((flow_outputs[j] + (- targets_copy[j])) ** 2 for j in range(len(targets_copy)))
                total_loss += loss.x
            avg_loss = total_loss / len(training_data)
            print(f"Iteration {idx}, Average Loss: {avg_loss:.4f}")
            test_circle_visual_ascii(nn)

    
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
    test_neuron_forward()
    test_neuron_forward_relu_negative()
    test_neural_net_flow()

    nn = NeuralNet([2, 8, 1])
    train_for_inside_circle(nn, iteraton_count=100, check_progress=True)
    test_circle_visual_ascii(nn)