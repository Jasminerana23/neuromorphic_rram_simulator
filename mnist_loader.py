"""
MNIST LOADER
Load and preprocess MNIST handwritten digit dataset
"""

import numpy as np
from urllib import request
import gzip
import os

class MNISTLoader:
    def __init__(self, data_dir='./mnist_data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_mnist(self):
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        files = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        
        print("📥 Downloading MNIST dataset...")
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            
            if os.path.exists(file_path):
                print(f"   ✅ {file} already exists")
                continue
            
            print(f"   Downloading {file}...")
            try:
                request.urlretrieve(base_url + file, file_path)
                print(f"   ✅ Downloaded {file}")
            except Exception as e:
                print(f"   ❌ Error downloading {file}: {e}")
    
    def load_images(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num, rows * cols)
            
            images = images / 255.0
            
            return images
    
    def load_labels(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
            
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
            return labels
    
    def one_hot_encode(self, labels, num_classes=10):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def get_training_data(self, normalize=True):
        images_path = os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')
        labels_path = os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')
        
        images = self.load_images(images_path)
        labels = self.load_labels(labels_path)
        labels_onehot = self.one_hot_encode(labels)
        
        return images, labels_onehot
    
    def get_test_data(self, normalize=True):
        images_path = os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')
        labels_path = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')
        
        images = self.load_images(images_path)
        labels = self.load_labels(labels_path)
        labels_onehot = self.one_hot_encode(labels)
        
        return images, labels_onehot
    
    def get_subset(self, X, y, num_samples):
        indices = np.random.choice(len(X), num_samples, replace=False)
        return X[indices], y[indices]
    
    def load_and_prepare(self, train_samples=None, test_samples=None):
        print("\n📊 MNIST Dataset Preparation")
        print("=" * 50)
        
        self.download_mnist()
        
        print("\n📂 Loading training data...")
        X_train, y_train = self.get_training_data()
        print(f"   ✅ Loaded {len(X_train)} training samples")
        
        print("\n📂 Loading test data...")
        X_test, y_test = self.get_test_data()
        print(f"   ✅ Loaded {len(X_test)} test samples")
        
        if train_samples:
            X_train, y_train = self.get_subset(X_train, y_train, train_samples)
            print(f"   ✅ Using {len(X_train)} training samples")
        
        if test_samples:
            X_test, y_test = self.get_subset(X_test, y_test, test_samples)
            print(f"   ✅ Using {len(X_test)} test samples")
        
        print("\n✅ Dataset ready!")
        print(f"   Training shape: {X_train.shape}")
        print(f"   Labels shape: {y_train.shape}")
        print(f"   Value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }


if __name__ == "__main__":
    print("=" * 60)
    print("MNIST LOADER TEST")
    print("=" * 60)
    
    loader = MNISTLoader('./mnist_data')
    
    print("\n✅ TEST 1: Load Small Subset")
    data = loader.load_and_prepare(train_samples=1000, test_samples=100)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   Training shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    print("\n✅ TEST 2: One-Hot Encoding")
    sample_label = y_train[0]
    digit_class = np.argmax(sample_label)
    print(f"   Sample one-hot: {sample_label}")
    print(f"   Decoded digit: {digit_class}")
    
    print("\n✅ TEST 3: Image Statistics")
    print(f"   Min value: {X_train.min():.4f}")
    print(f"   Max value: {X_train.max():.4f}")
    print(f"   Mean value: {X_train.mean():.4f}")
    print(f"   Std value: {X_train.std():.4f}")
    
    print("\n✅ TEST 4: Data Shapes")
    print(f"   Images: {X_train.shape} (samples, pixels)")
    print(f"   Labels: {y_train.shape} (samples, classes)")
    print(f"   Each image: 784 pixels (28×28)")
    print(f"   Classes: 10 (0-9)")
    
    print("\n✅ MNIST loader test complete!")
