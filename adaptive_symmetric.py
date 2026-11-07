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

def quantize_sym(W, L):
    Wmax = np.abs(W).max()
    return Wmax * np.round(W / Wmax * (L - 1)) / (L - 1)

def run_seed(seed):
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    Wq = {L: [quantize_sym(w, L) for w in W] for L in [64, 128, 256]}
    P_va, P_te = {}, {}
    for L in [64, 128, 256]:
        pv, _ = forward(X_va, Wq[L], b)
        pt, _ = forward(X_te, Wq[L], b)
        P_va[L] = pv[-1]
        P_te[L] = pt[-1]

    conf_va = P_va[64].max(axis=1)
    conf_te = P_te[64].max(axis=1)

    def adaptive(conf, P, Te, Tm):
        tier = np.where(conf >= Te, 0, np.where(conf >= Tm, 1, 2))
        pred = np.where(tier == 0, np.argmax(P[64], 1),
               np.where(tier == 1, np.argmax(P[128], 1), np.argmax(P[256], 1)))
        return pred, np.array([64, 128, 256])[tier]

    best = None
    for Te in [0.80, 0.85, 0.90, 0.95, 0.99]:
        for Tm in [0.40, 0.50, 0.60, 0.70, 0.80]:
            if Tm >= Te:
                continue
            pred, lv = adaptive(conf_va, P_va, Te, Tm)
            key = (np.mean(pred == y_va), -lv.mean())
            if best is None or key > best[0]:
                best = (key, Te, Tm)
    _, Te, Tm = best

    pred, lv = adaptive(conf_te, P_te, Te, Tm)
    fixed = {L: np.mean(np.argmax(P_te[L], 1) == y_te) for L in [64, 128, 256]}
    return fixed[64], fixed[128], fixed[256], np.mean(pred == y_te), lv.mean(), Te, Tm

rows = []
for s in range(5):
    r = run_seed(s)
    rows.append(r)
    print(f"seed {s}: fix64={r[0]:.4f} fix128={r[1]:.4f} fix256={r[2]:.4f} "
          f"adapt={r[3]:.4f} avg_lvl={r[4]:.1f} thr=({r[5]},{r[6]})")

R = np.array([r[:5] for r in rows])
names = ["Fixed 64", "Fixed 128", "Fixed 256", "Adaptive", "Avg levels"]
print("=" * 70)
print("MEAN ± STD over 5 seeds (symmetric quantizer, test set)")
for j, name in enumerate(names):
    if j < 4:
        print(f"  {name:12s}: {R[:, j].mean():.4f} ± {R[:, j].std():.4f}")
    else:
        print(f"  {name:12s}: {R[:, j].mean():.1f} ± {R[:, j].std():.1f}")
print("=" * 70)