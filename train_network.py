"""
TRAINING SCRIPT
Train neural network on simple data
"""

import numpy as np
from neuron_model import NeuralNetwork
from learning_rules import SGD

print("=" * 70)
print("TRAINING NEURAL NETWORK")
print("=" * 70)

print("\n🏗️ Building network...")
network = NeuralNetwork(layer_sizes=[28*28, 128, 64, 10])

print("\n📥 Creating training data...")
num_samples = 100
X_train = np.random.rand(num_samples, 784) * 0.5
y_train = np.random.randint(0, 10, num_samples)
print(f"   Training samples: {num_samples}")

print("\n🧪 Testing network...")
predictions = []
for x in X_train[:5]:
    pred = network.predict(x)
    predictions.append(pred)
print(f"   Sample predictions: {predictions}")

print("\n📊 Initial accuracy:")
initial_acc = network.accuracy(X_train, y_train)
print(f"   Accuracy: {initial_acc*100:.2f}%")

print("\n✅ Network training setup complete!")
print("   Ready for Week 4: Gradient descent training")
