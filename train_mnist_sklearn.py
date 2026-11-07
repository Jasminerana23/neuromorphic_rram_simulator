"""
TRAIN MNIST - SKLEARN VERSION
Complete training pipeline using sklearn MNIST loader (NO TensorFlow issues!)
"""

import numpy as np
import time
from neuron_model import NeuralNetwork
from backpropagation import Backpropagation
from mnist_loader_sklearn import MNISTLoader

def simple_train(network, X_train, y_train, X_val, y_val, epochs=20, learning_rate=0.01):
    """
    Simple training loop for Week 4
    """
    
    backprop = Backpropagation(learning_rate)
    
    print("\n" + "=" * 70)
    print("MNIST TRAINING - WEEK 4 (SKLEARN VERSION)")
    print("=" * 70)
    print(f"\n🧠 Training Configuration:")
    print(f"   Network: 64 → 128 → 64 → 10")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"\n🚀 Starting training...\n")
    
    best_accuracy = 0
    accuracies = []
    losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        epoch_loss = 0
        correct = 0
        
        # Training loop
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            # Forward pass
            output = network.forward(x)
            
            # Calculate loss
            loss = backprop.calculate_loss(output, y)
            epoch_loss += loss
            
            # Check if correct
            pred_class = np.argmax(output)
            true_class = np.argmax(y)
            if pred_class == true_class:
                correct += 1
            
            # Simple weight update
            error = output - y
            for layer in network.layers:
                for neuron in layer.neurons:
                    gradient = error * neuron.last_input
                    neuron.weights -= learning_rate * gradient * 0.01
        
        # Average metrics
        epoch_loss = epoch_loss / len(X_train)
        train_acc = correct / len(X_train)
        losses.append(epoch_loss)
        
        # Validation
        val_correct = 0
        for x, y in zip(X_val, y_val):
            output = network.forward(x)
            pred_class = np.argmax(output)
            true_class = np.argmax(y)
            if pred_class == true_class:
                val_correct += 1
        
        val_acc = val_correct / len(X_val)
        accuracies.append(val_acc)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {train_acc*100:5.2f}% | "
                  f"Val Acc: {val_acc*100:5.2f}% | "
                  f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Final Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Best Validation Accuracy: {best_accuracy*100:.2f}%")
    print(f"   Total Training Time: {total_time:.2f} seconds")
    print(f"   Average Time per Epoch: {total_time/epochs:.2f}s")
    
    if len(accuracies) > 1:
        improvement = (accuracies[-1] - accuracies[0]) * 100
        print(f"\n📈 Accuracy Improvement:")
        print(f"   Initial: {accuracies[0]*100:.2f}%")
        print(f"   Final: {accuracies[-1]*100:.2f}%")
        print(f"   Improvement: +{improvement:.2f}%")
    
    return {
        'network': network,
        'losses': losses,
        'accuracies': accuracies,
        'best_accuracy': best_accuracy
    }


def evaluate_on_test(network, X_test, y_test):
    """Evaluate trained network on test dataset"""
    
    print("\n" + "=" * 70)
    print("TESTING ON TEST DATASET")
    print("=" * 70)
    
    correct = 0
    
    for x, y in zip(X_test, y_test):
        output = network.forward(x)
        pred_class = np.argmax(output)
        true_class = np.argmax(y)
        
        if pred_class == true_class:
            correct += 1
    
    test_accuracy = correct / len(X_test)
    
    print(f"\n📋 Test Results:")
    print(f"   Total test samples: {len(X_test)}")
    print(f"   Correct predictions: {correct}/{len(X_test)}")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    
    if test_accuracy > 0.90:
        print(f"\n🎉 EXCELLENT! Accuracy > 90%!")
    elif test_accuracy > 0.80:
        print(f"\n✅ GOOD! Accuracy > 80%!")
    elif test_accuracy > 0.70:
        print(f"\n⏳ DECENT! Keep improving!")
    else:
        print(f"\n📈 Continue training in Week 5!")
    
    return test_accuracy


# ================================================
# MAIN EXECUTION
# ================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("WEEK 4: MNIST TRAINING PIPELINE (SKLEARN VERSION)")
    print("=" * 70)
    
    # STEP 1: Load MNIST dataset
    print("\n[STEP 1] Loading MNIST Dataset...")
    loader = MNISTLoader()
    
    # Load all data
    print("\nLoading data...")
    data = loader.load_and_prepare(train_samples=None, test_samples=None)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split training into train/validation (80/20)
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"\n   Train: {len(X_train_split)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]} (image size)")
    
    # STEP 2: Create neural network
    print("\n[STEP 2] Creating Neural Network...")
    # Input: 64 features (8×8 images from sklearn)
    network = NeuralNetwork(layer_sizes=[64, 128, 64, 10])
    
    # STEP 3: Train
    print("\n[STEP 3] Training Network...")
    results = simple_train(
        network=network,
        X_train=X_train_split,
        y_train=y_train_split,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
        learning_rate=0.01
    )
    
    # STEP 4: Evaluate on test set
    print("\n[STEP 4] Evaluating on Test Set...")
    test_accuracy = evaluate_on_test(network, X_test, y_test)
    
    # STEP 5: Summary
    print("\n" + "=" * 70)
    print("WEEK 4 SUMMARY")
    print("=" * 70)
    print(f"\n✅ Network trained successfully!")
    print(f"   Validation accuracy: {results['best_accuracy']*100:.2f}%")
    print(f"   Test accuracy: {test_accuracy*100:.2f}%")
    
    if test_accuracy > 0.90:
        print(f"\n🎉 WEEK 4 TARGET ACHIEVED! (>90% accuracy)")
    else:
        print(f"\n📈 Continue training in Week 5 for better accuracy")
    
    print("\n✅ Week 4 training complete!")
    print("   Ready to commit to GitHub! 🚀")
