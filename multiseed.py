import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0

def onehot(labels):
    Y = np.zeros((len(labels), 10))
    Y[np.arange(len(labels)), labels] = 1
    return Y

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

def train(X_tr, y_tr, W, b, epochs=50, lr=0.1, batch=32):
    Ys = onehot(y_tr)
    for ep in range(epochs):
        idx = np.random.permutation(len(X_tr))
        for s in range(0, len(X_tr), batch):
            bi = idx[s:s+batch]
            xb, yb = X_tr[bi], Ys[bi]
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

def run_one(seed):
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    Wq = {L: [quantize(w, L) for w in W] for L in [64, 128, 256]}

    # Fixed-precision accuracies on test
    fixed = {}
    pred_te = {}
    for L in [64, 128, 256]:
        p, _ = forward(X_te, Wq[L], b)
        pred_te[L] = np.argmax(p[-1], axis=1)
        fixed[L] = np.mean(pred_te[L] == y_te)

    # Thresholds tuned on VALIDATION (same grid as before)
    pv64, _ = forward(X_va, Wq[64], b)
    conf_va = pv64[-1].max(axis=1)
    pv = {}
    for L in [64, 128, 256]:
        p, _ = forward(X_va, Wq[L], b)
        pv[L] = np.argmax(p[-1], axis=1)

    def adaptive(conf, pred, Te, Tm):
        tier = np.where(conf >= Te, 0, np.where(conf >= Tm, 1, 2))
        p = np.where(tier == 0, pred[64], np.where(tier == 1, pred[128], pred[256]))
        return p, np.array([64, 128, 256])[tier]

    best = None
    for Te in [0.80, 0.85, 0.90, 0.95, 0.99]:
        for Tm in [0.40, 0.50, 0.60, 0.70, 0.80]:
            if Tm >= Te:
                continue
            p, lv = adaptive(conf_va, pv, Te, Tm)
            key = (np.mean(p == y_va), -lv.mean())
            if best is None or key > best[0]:
                best = (key, Te, Tm)
    _, Te, Tm = best

    # Test with validation-chosen thresholds (once)
    pt64, _ = forward(X_te, Wq[64], b)
    conf_te = pt64[-1].max(axis=1)
    p, lv = adaptive(conf_te, pred_te, Te, Tm)
    adapt_acc = np.mean(p == y_te)

    return fixed[64], fixed[128], fixed[256], adapt_acc, lv.mean(), Te, Tm

seeds = [0, 1, 2, 3, 4]
rows = []
for s in seeds:
    r = run_one(s)
    rows.append(r)
    print(f"seed {s}: fix64={r[0]:.4f} fix128={r[1]:.4f} fix256={r[2]:.4f} "
          f"adapt={r[3]:.4f} avg_lvl={r[4]:.1f} thr=({r[5]},{r[6]})")

R = np.array([r[:5] for r in rows])
print("=" * 70)
print("MEAN ± STD over 5 seeds (test set)")
names = ["Fixed 64", "Fixed 128", "Fixed 256", "Adaptive", "Avg levels"]
for j, name in enumerate(names):
    print(f"  {name:12s}: {R[:, j].mean():.4f} ± {R[:, j].std():.4f}"
          if j < 4 else f"  {name:12s}: {R[:, j].mean():.1f} ± {R[:, j].std():.1f}")
print("=" * 70)