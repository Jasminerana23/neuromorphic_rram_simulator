"""
LEARNING RULES
Implements training algorithms for neural networks
"""

import numpy as np

class TrainingRule:
    """Base class for training rules"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_weights(self, weights, gradients):
        raise NotImplementedError


class SGD(TrainingRule):
    """Stochastic Gradient Descent"""
    
    def update_weights(self, weights, gradients):
        return weights - self.learning_rate * gradients


class HebbianRule(TrainingRule):
    """Hebbian learning rule"""
    
    def update_weights(self, weights, pre_activation, post_activation):
        delta_w = self.learning_rate * np.outer(pre_activation, post_activation)
        return weights + delta_w


class STDPRule(TrainingRule):
    """Spike-Timing Dependent Plasticity"""
    
    def __init__(self, learning_rate=0.01, tau=20):
        super().__init__(learning_rate)
        self.tau = tau
    
    def update_weights(self, weights, time_diff):
        if time_diff > 0:
            delta_w = self.learning_rate * np.exp(-time_diff / self.tau)
        else:
            delta_w = -self.learning_rate * np.exp(time_diff / self.tau)
        
        return weights + delta_w


if __name__ == "__main__":
    print("=" * 60)
    print("LEARNING RULES TEST")
    print("=" * 60)
    
    print("\n✅ TEST 1: Stochastic Gradient Descent")
    sgd = SGD(learning_rate=0.01)
    weights = np.array([0.5, 0.3, 0.8])
    gradients = np.array([0.1, 0.2, 0.05])
    updated = sgd.update_weights(weights, gradients)
    print(f"Before: {weights}")
    print(f"After:  {updated}")
    
    print("\n✅ TEST 2: Hebbian Learning")
    hebbian = HebbianRule(learning_rate=0.01)
    pre = np.array([0.5, 0.3, 0.8])
    post = 0.7
    updated = hebbian.update_weights(weights, pre, post)
    print(f"Before: {weights}")
    print(f"After:  {updated}")
    
    print("\n✅ TEST 3: STDP Learning")
    stdp = STDPRule(learning_rate=0.01, tau=20)
    time_diff = 10
    updated = stdp.update_weights(weights, time_diff)
    print(f"Before: {weights}")
    print(f"After:  {updated}")
    
    print("\n✅ All learning rule tests complete!")
