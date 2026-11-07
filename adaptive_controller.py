import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ---- Data (same split as before) ----
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

def onehot(labels):
    Y = np.zeros((len(labels), 10))
    Y[np.arange(len(labels)), labels] = 1
    return Y

# ---- Network: 64 -> 128 -> 64 -> 10 ----
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
    w_min, w_max = w.min(), w.max()
    step = (w_max - w_min) / (levels - 1)
    return np.round((w - w_min) / step) * step + w_min

# ---- Train float model ----
train()

# ---- Fixed-precision models ----
W64  = [quantize(w, 64) for w in W]
W128 = [quantize(w, 128) for w in W]
W256 = [quantize(w, 256) for w in W]

# ---- Difficulty signal: confidence (max softmax prob) from the 64-level model ----
# Easy = high confidence -> use 64 levels
# Medium = medium confidence -> 128 levels
# Hard = low confidence -> 256 levels
p64, _ = forward(X_te, W64, b)
conf = p64[-1].max(axis=1)

T_easy, T_med = 0.90, 0.60   # thresholds (to be tuned on validation set)
tier = np.where(conf >= T_easy, 0, np.where(conf >= T_med, 1, 2))

# ---- Prediction per sample, using the chosen tier ----
pred_fixed = {
    64:  np.argmax(p64[-1], axis=1),
    128: np.argmax(forward(X_te, W128, b)[0][-1], axis=1),
    256: np.argmax(forward(X_te, W256, b)[0][-1], axis=1),
}
levels_of_tier = np.array([64, 128, 256])
pred_adapt = np.where(tier == 0, pred_fixed[64],
             np.where(tier == 1, pred_fixed[128], pred_fixed[256]))

# ---- Results ----
print("=" * 60)
print("FIXED PRECISION")
for L in [64, 128, 256]:
    print(f"  {L:3d} levels : {np.mean(pred_fixed[L] == y_te):.4f}")

print("=" * 60)
print("ADAPTIVE PRECISION (64/128/256 by confidence)")
print(f"  Accuracy      : {np.mean(pred_adapt == y_te):.4f}")
print(f"  Avg levels    : {levels_of_tier[tier].mean():.1f}")
for t, name in enumerate(["easy->64", "medium->128", "hard->256"]):
    cnt = np.sum(tier == t)
    print(f"  {name:12s}: {cnt:3d} samples ({cnt/len(tier):.1%})")
print("=" * 60)