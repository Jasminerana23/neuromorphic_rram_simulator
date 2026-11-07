"""
MNIST LOADER - SKLEARN VERSION
Load MNIST using sklearn (Simple, reliable, NO TensorFlow issues!)
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class MNISTLoader:
    """
    Load MNIST dataset using sklearn
    Simple, fast, NO compatibility issues!
    """
    
    def __init__(self):
        print("✅ MNIST Loader initialized (sklearn version)")
    
    def one_hot_encode(self, labels, num_classes=10):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def load_data(self):
        """
        Load MNIST dataset using sklearn
        
        sklearn digits: 1797 samples, 8×8 images (64 features)
        Easy to load, NO download needed!
        """
        print("\n📊 MNIST Dataset Loading")
        print("=" * 50)
        
        print("\n📥 Loading MNIST from sklearn...")
        
        # Load digits dataset (built-in)
        digits = load_digits()
        X = digits.data  # (1797, 64) - 8×8 images flattened
        y = digits.target  # (1797,) - labels 0-9
        
        print(f"   ✅ Loaded!")
        print(f"   Data shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Normalize to [0, 1]
        print("\n🔄 Normalizing data...")
        X = X / 16.0  # sklearn digits are 0-16, normalize to 0-1
        
        # Shuffle and split
        print("   Shuffling data...")
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split into train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # One-hot encode
        print("   One-hot encoding labels...")
        y_train_onehot = self.one_hot_encode(y_train, num_classes=10)
        y_test_onehot = self.one_hot_encode(y_test, num_classes=10)
        
        print("\n✅ Dataset ready!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features per image: {X_train.shape[1]}")
        print(f"   Value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        
        return {
            'X_train': X_train,
            'y_train': y_train_onehot,
            'X_test': X_test,
            'y_test': y_test_onehot
        }
    
    def get_subset(self, X, y, num_samples):
        """Get random subset of data"""
        if num_samples >= len(X):
            return X, y
        indices = np.random.choice(len(X), num_samples, replace=False)
        return X[indices], y[indices]
    
    def load_and_prepare(self, train_samples=None, test_samples=None):
        """
        Complete pipeline with optional subsets
        """
        data = self.load_data()
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Optional subsets
        if train_samples:
            X_train, y_train = self.get_subset(X_train, y_train, train_samples)
            print(f"   📌 Using {len(X_train)} training samples")
        
        if test_samples:
            X_test, y_test = self.get_subset(X_test, y_test, test_samples)
            print(f"   📌 Using {len(X_test)} test samples")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }


# ================================================
# TEST CODE
# ================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MNIST LOADER TEST (SKLEARN VERSION)")
    print("=" * 60)
    
    loader = MNISTLoader()
    
    print("\n✅ TEST 1: Load Full Dataset")
    data = loader.load_and_prepare()
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   ✅ Training shape: {X_train.shape}")
    print(f"   ✅ Test shape: {X_test.shape}")
    
    print("\n✅ TEST 2: Load Subset")
    data_subset = loader.load_and_prepare(train_samples=100, test_samples=30)
    
    print(f"   ✅ Subset training shape: {data_subset['X_train'].shape}")
    print(f"   ✅ Subset test shape: {data_subset['X_test'].shape}")
    
    print("\n✅ TEST 3: One-Hot Encoding Check")
    sample_label = y_train[0]
    digit_class = np.argmax(sample_label)
    print(f"   One-hot sample: {sample_label}")
    print(f"   Decoded digit: {digit_class}")
    
    print("\n✅ TEST 4: Data Statistics")
    print(f"   Min value: {X_train.min():.4f}")
    print(f"   Max value: {X_train.max():.4f}")
    print(f"   Mean value: {X_train.mean():.4f}")
    print(f"   Std value: {X_train.std():.4f}")
    
    print("\n✅ TEST 5: Random Sample")
    sample_idx = np.random.randint(0, len(X_train))
    sample_image = X_train[sample_idx]
    sample_digit = np.argmax(y_train[sample_idx])
    print(f"   Sample image shape: {sample_image.shape}")
    print(f"   Sample digit: {sample_digit}")
    
    print("\n✅ MNIST loader test COMPLETE!")
    print("   Ready for training! 🚀")
