from neural_net import Neuron, NeuralNet
from autograd_backward import Node

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

if __name__ == "__main__":
    test_neuron_forward()
    test_neuron_forward_relu_negative()
    test_neural_net_flow()