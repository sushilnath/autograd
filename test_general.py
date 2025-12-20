import torch
import torchvision
import torchvision.transforms as transforms
from neural_net import Node, NeuralNet

def load_mnist_data():
    print("Fetching MNIST Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize based on the average and mean color intensity of MNIST.
    ])

    # Training dataset.
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
    
    # Displaying a sample image from the dataset and its label.
    sample_image, sample_label = train_dataset[0]
    print(f"Sample Image Shape: {sample_image.shape}, Sample Label: {sample_label}")
    display_image_ascii_and_label(sample_image, sample_label)

    # Network details.
    input_size = 28 * 28 # MNIST images are 28x28 pixels.
    hidden_size = 16
    output_size = 10 # 10 classes for digits 0-9.
    
    model = NeuralNet([input_size, hidden_size, output_size])
    
    iteration_count = 1

    for epoch in range(iteration_count):
        print(f"Epoch {epoch + 1}/{iteration_count}")
        for i in range(len(train_dataset)):
            image, label = train_dataset[i]
            # image = image.view(-1) # Flatten the image to a vector of size 784.
            inputs = [Node(pixel) for pixel in image.view(-1).tolist()]
            targets = [Node(1.0) if j == label else Node(0.0) for j in range(10)]
            model.train(inputs, targets, learning_rate=0.01)
        
        print(f"Completed Epoch {epoch + 1}")

def display_image_ascii_and_label(image_tensor, label):
    print(f"Label: {label}")
    for i in range(28):
        line = ""
        for j in range(28):
            pixel_value = image_tensor[0, i, j].item()
            if pixel_value > 0.5:
                line += "#"
            elif pixel_value > 0.2:
                line += "*"
            else:
                line += " "
        print(line)

if __name__ == "__main__":
    load_mnist_data()