"""
Hello World: 128×128 Crossbar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossbar_array import CrossbarArray
import numpy as np

print("=" * 70)
print("TESTING 128×128 CROSSBAR (16,384 RRAMs)")
print("=" * 70)

# Create 128×128 crossbar
print("\n🏗️ Building 128×128 crossbar...")
crossbar = CrossbarArray(rows=128, cols=128)

# Random input
print("\n📥 Creating random input...")
inputs = np.random.rand(128) * 0.5

# Forward pass
print("\n🔄 Forward pass...")
outputs = crossbar.forward(list(inputs))
print(f"Output shape: {len(outputs)}")
print(f"Output range: [{min(outputs):.6f}, {max(outputs):.6f}]")

# Learning
print("\n🧠 Learning (3 epochs)...")
for epoch in range(3):
    outputs = crossbar.forward(list(inputs))
    crossbar.hebbian_learn(list(inputs), outputs, learning_rate=0.1)
    print(f"  Epoch {epoch+1}: Complete")

# Statistics
print("\n📊 Statistics:")
stats = crossbar.get_statistics()
print(f"Mean resistance: {stats['mean_resistance']:.2f} Ω")
print(f"Std resistance: {stats['std_resistance']:.2f} Ω")

print("\n✅ 128×128 CROSSBAR WORKING!")
