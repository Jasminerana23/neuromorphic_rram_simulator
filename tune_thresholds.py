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

train()
W64  = [quantize(w, 64) for w in W]
W128 = [quantize(w, 128) for w in W]
W256 = [quantize(w, 256) for w in W]

# Confidence on VALIDATION set
p_va64, _ = forward(X_va, W64, b)
p_va128, _ = forward(X_va, W128, b)
p_va256, _ = forward(X_va, W256, b)
conf_va = p_va64[-1].max(axis=1)
pred_va = {
    64:  np.argmax(p_va64[-1], axis=1),
    128: np.argmax(p_va128[-1], axis=1),
    256: np.argmax(p_va256[-1], axis=1),
}

def run_adaptive(conf, pred, T_easy, T_med, y_true):
    tier = np.where(conf >= T_easy, 0, np.where(conf >= T_med, 1, 2))
    p = np.where(tier == 0, pred[64], np.where(tier == 1, pred[128], pred[256]))
    levels = np.array([64, 128, 256])[tier]
    return np.mean(p == y_true), levels.mean(), tier

# Grid search on VALIDATION only
best = None
print("Grid search on VALIDATION set")
print(f"{'T_easy':>7} {'T_med':>7} {'val_acc':>9} {'avg_lvl':>9}")
for T_easy in [0.80, 0.85, 0.90, 0.95, 0.99]:
    for T_med in [0.40, 0.50, 0.60, 0.70, 0.80]:
        if T_med >= T_easy:
            continue
        acc, avg, _ = run_adaptive(conf_va, pred_va, T_easy, T_med, y_va)
        print(f"{T_easy:7.2f} {T_med:7.2f} {acc:9.4f} {avg:9.1f}")
        # Pick: highest val acc; tie-break = fewer average levels
        key = (acc, -avg)
        if best is None or key > best[0]:
            best = (key, T_easy, T_med)

_, bE, bM = best
print("=" * 60)
print(f"Chosen thresholds (validation): T_easy={bE}, T_med={bM}")

# ONE-TIME test evaluation with chosen thresholds
p_te64, _ = forward(X_te, W64, b)
p_te128, _ = forward(X_te, W128, b)
p_te256, _ = forward(X_te, W256, b)
conf_te = p_te64[-1].max(axis=1)
pred_te = {
    64:  np.argmax(p_te64[-1], axis=1),
    128: np.argmax(p_te128[-1], axis=1),
    256: np.argmax(p_te256[-1], axis=1),
}
acc, avg, tier = run_adaptive(conf_te, pred_te, bE, bM, y_te)
print("TEST (one-time, using validation-chosen thresholds)")
print(f"  Adaptive accuracy : {acc:.4f}")
print(f"  Avg levels        : {avg:.1f}")
for t, name in enumerate(["easy->64", "medium->128", "hard->256"]):
    cnt = np.sum(tier == t)
    print(f"  {name:12s}: {cnt:3d} samples ({cnt/len(tier):.1%})")
print("=" * 60)