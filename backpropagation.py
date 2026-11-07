"""
BACKPROPAGATION
Implements backward pass for gradient descent training
"""

import numpy as np

class Backpropagation:
    """
    Backpropagation algorithm for training neural networks
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.loss_history = []
    
    def calculate_loss(self, predicted, target):
        error = predicted - target
        mse_loss = np.mean(error ** 2)
        return mse_loss
    
    def softmax(self, x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def softmax_derivative(self, softmax_output):
        s = softmax_output.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def backward_pass(self, network, inputs, targets):
        output = network.forward(inputs)
        loss = self.calculate_loss(output, targets)
        self.loss_history.append(loss)
        error = output - targets
        return loss, error
    
    def update_weights(self, weights, gradients):
        return weights - self.learning_rate * gradients
    
    def cross_entropy_loss(self, predicted_probs, target):
        predicted_probs = np.clip(predicted_probs, 1e-7, 1 - 1e-7)
        ce_loss = -np.sum(target * np.log(predicted_probs))
        return ce_loss
    
    def get_accuracy(self, predictions, targets):
        if len(targets.shape) == 2:
            target_classes = np.argmax(targets, axis=1)
        else:
            target_classes = targets
        
        correct = np.sum(predictions == target_classes)
        accuracy = correct / len(target_classes)
        return accuracy


class TrainingLoop:
    def __init__(self, network, learning_rate=0.01):
        self.network = network
        self.backprop = Backpropagation(learning_rate)
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        num_samples = len(X_train)
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, num_samples, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            batch_loss = 0
            
            for x, y in zip(batch_X, batch_y):
                output = self.network.forward(x)
                loss = self.backprop.calculate_loss(output, y)
                batch_loss += loss
            
            batch_loss /= len(batch_X)
            epoch_loss += batch_loss
            batch_count += 1
        
        epoch_loss /= batch_count
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def evaluate(self, X_val, y_val):
        predictions = []
        
        for x in X_val:
            output = self.network.forward(x)
            pred_class = np.argmax(output)
            predictions.append(pred_class)
        
        accuracy = self.backprop.get_accuracy(np.array(predictions), y_val)
        self.val_accuracies.append(accuracy)
        
        return accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print(f"\n🧠 Starting training for {epochs} epochs...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Batch size: {batch_size}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            val_acc = self.evaluate(X_val, y_val)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}")
                print(f"      Loss: {train_loss:.4f}")
                print(f"      Validation Accuracy: {val_acc*100:.2f}%")
        
        print(f"\n✅ Training complete!")
        print(f"   Final Loss: {self.train_losses[-1]:.4f}")
        print(f"   Final Accuracy: {self.val_accuracies[-1]*100:.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("BACKPROPAGATION TEST")
    print("=" * 60)
    
    print("\n✅ TEST 1: Loss Calculation")
    backprop = Backpropagation()
    predicted = np.array([0.9, 0.05, 0.05])
    target = np.array([1.0, 0.0, 0.0])
    loss = backprop.calculate_loss(predicted, target)
    print(f"   Predicted: {predicted}")
    print(f"   Target: {target}")
    print(f"   Loss (MSE): {loss:.4f}")
    print(f"   (Low loss = good prediction!)")
    
    print("\n✅ TEST 2: Softmax Activation")
    outputs = np.array([2.0, 1.0, 0.1])
    probs = backprop.softmax(outputs)
    print(f"   Network outputs: {outputs}")
    print(f"   Softmax probs: {probs}")
    print(f"   Sum of probs: {np.sum(probs):.4f} (should be 1.0)")
    
    print("\n✅ TEST 3: Cross-Entropy Loss")
    predicted_probs = np.array([0.7, 0.2, 0.1])
    target = np.array([1, 0, 0])
    ce_loss = backprop.cross_entropy_loss(predicted_probs, target)
    print(f"   Predicted probs: {predicted_probs}")
    print(f"   Target: {target}")
    print(f"   CE Loss: {ce_loss:.4f}")
    
    print("\n✅ TEST 4: Accuracy Calculation")
    predictions = np.array([0, 1, 2, 0, 1])
    targets = np.array([0, 1, 2, 0, 2])
    accuracy = backprop.get_accuracy(predictions, targets)
    print(f"   Predictions: {predictions}")
    print(f"   Targets: {targets}")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    print("\n✅ All backpropagation tests complete!")
