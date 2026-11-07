"""
NEURON MODEL
Implements neurons with activation functions for training
"""

import numpy as np

class Neuron:
    """Single neuron with activation function"""
    
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0
        self.last_input = None
        self.last_output = None
    
    def forward(self, inputs):
        self.last_input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.last_output = self.relu(z)
        return self.last_output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class NeuronLayer:
    """Layer of neurons"""
    
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return np.array(outputs)
    
    def get_weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
        return np.array(weights)


class NeuralNetwork:
    """Multi-layer neural network for training"""
    
    def __init__(self, layer_sizes):
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            layer = NeuronLayer(layer_sizes[i+1], layer_sizes[i])
            self.layers.append(layer)
        
        print(f"✅ Network created: {' → '.join(map(str, layer_sizes))}")
    
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs)
    
    def accuracy(self, inputs_list, labels_list):
        correct = 0
        for inputs, true_label in zip(inputs_list, labels_list):
            prediction = self.predict(inputs)
            if prediction == true_label:
                correct += 1
        
        accuracy = correct / len(inputs_list)
        return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("NEURON MODEL TEST")
    print("=" * 60)
    
    print("\n✅ TEST 1: Single Neuron")
    neuron = Neuron(num_inputs=5)
    inputs = np.random.rand(5) * 0.5
    output = neuron.forward(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output: {output}")
    
    print("\n✅ TEST 2: Neuron Layer (10 neurons)")
    layer = NeuronLayer(num_neurons=10, num_inputs=5)
    outputs = layer.forward(inputs)
    print(f"Layer output shape: {outputs.shape}")
    print(f"Mean output: {np.mean(outputs):.4f}")
    
    print("\n✅ TEST 3: Multi-layer Network")
    network = NeuralNetwork(layer_sizes=[784, 128, 64, 10])
    
    test_input = np.random.rand(784) * 0.5
    prediction = network.predict(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {network.forward(test_input)[prediction]:.4f}")
    
    print("\n✅ All neuron tests complete!")
