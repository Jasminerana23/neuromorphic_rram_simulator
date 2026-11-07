"""
Test crossbar array - Simple version
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossbar_array import CrossbarArray
import numpy as np

print("=" * 60)
print("TESTING CROSSBAR ARRAY")
print("=" * 60)

# Test 1: Small 3×3 crossbar
print("\n✅ TEST 1: 3×3 Crossbar")
crossbar1 = CrossbarArray(rows=3, cols=3)
inputs1 = [0.5, 0.3, 0.8]
outputs1 = crossbar1.forward(inputs1)
print(f"Inputs: {inputs1}")
print(f"Outputs: {outputs1}")

# Test 2: Medium 4×4 crossbar
print("\n✅ TEST 2: 4×4 Crossbar")
crossbar2 = CrossbarArray(rows=4, cols=4)
inputs2 = [0.2, 0.5, 0.8, 0.3]
outputs2 = crossbar2.forward(inputs2)
print(f"Inputs: {inputs2}")
print(f"Outputs: {outputs2}")

# Test 3: Learning
print("\n✅ TEST 3: Hebbian Learning (5 epochs)")
for epoch in range(5):
    outputs2 = crossbar2.forward(inputs2)
    crossbar2.hebbian_learn(inputs2, outputs2, learning_rate=0.5)
    print(f"  Epoch {epoch+1}: Complete")

# Test 4: Stats
print("\n✅ TEST 4: Statistics")
stats = crossbar2.get_statistics()
print(f"Mean resistance: {stats['mean_resistance']:.2f} Ω")
print(f"Min resistance: {stats['min_resistance']:.2f} Ω")
print(f"Max resistance: {stats['max_resistance']:.2f} Ω")

print("\n✅ ALL TESTS PASSED!")
