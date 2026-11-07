import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

def onehot(labels):
    Y = np.zeros((len(labels), 10))
    Y[np.arange(len(labels)), labels] = 1
    return Y

sizes = [64, 128, 64, 10]
W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
b = [np.zeros(sizes[i+1]) for i in range(3)]

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W, b):
    acts, zs = [X], []
    a = X
    for i in range(3):
        z = a @ W[i] + b[i]
        zs.append(z)
        a = np.maximum(0, z) if i < 2 else softmax(z)
        acts.append(a)
    return acts, zs

def accuracy(X, y, W, b):
    acts, _ = forward(X, W, b)
    return np.mean(np.argmax(acts[-1], axis=1) == y)

def train(epochs=50, lr=0.1, batch=32):
    Xs, Ys = X_tr, onehot(y_tr)
    for ep in range(epochs):
        idx = np.random.permutation(len(Xs))
        for s in range(0, len(Xs), batch):
            bi = idx[s:s+batch]
            xb, yb = Xs[bi], Ys[bi]
            acts, zs = forward(xb, W, b)
            n = len(bi)
            delta = (acts[-1] - yb) / n
            for i in reversed(range(3)):
                dW = acts[i].T @ delta
                db = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ W[i].T) * (zs[i-1] > 0)
                W[i] -= lr * dW
                b[i] -= lr * db

def quantize(w, levels):
    # Uniform quantization of weights into 'levels' discrete values
    w_min, w_max = w.min(), w.max()
    step = (w_max - w_min) / (levels - 1)
    return np.round((w - w_min) / step) * step + w_min

train()
print("=" * 60)
print("FLOAT (unquantized) TEST ACC:", round(accuracy(X_te, y_te, W, b), 4))
print("=" * 60)

for levels in [64, 128, 256]:
    Wq = [quantize(w, levels) for w in W]
    bq = b
    acc = accuracy(X_te, y_te, Wq, bq)
    print(f"Fixed {levels:3d} levels | Test acc: {acc:.4f}")

print("=" * 60)