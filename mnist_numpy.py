import numpy as np
from sklearn.datasets import fetch_openml

np.random.seed(0)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float64) / 255.0
y = y.astype(int)

X_tr_full, X_te, y_tr_full, y_te = X[:60000], X[60000:], y[:60000], y[60000:]
X_tr, X_va, y_tr, y_va = X_tr_full[:55000], X_tr_full[55000:], y_tr_full[:55000], y_tr_full[55000:]
print(f"Train {len(X_tr)}, Val {len(X_va)}, Test {len(X_te)}")

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

def train(X_tr, y_tr, W, b, epochs=10, lr=0.1, batch=64):
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
        va_acc = np.mean(np.argmax(forward(X_va, W, b)[0][-1], 1) == y_va)
        print(f"Epoch {ep+1:2d} | Val acc: {va_acc:.4f}")

def quantize_sym(W, L):
    Wmax = np.abs(W).max()
    return Wmax * np.round(W / Wmax * (L - 1)) / (L - 1)

sizes = [784, 128, 64, 10]
W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
b = [np.zeros(sizes[i+1]) for i in range(3)]
train(X_tr, y_tr, W, b)

P_va, P_te = {}, {}
Wq = {L: [quantize_sym(w, L) for w in W] for L in [64, 128, 256]}
for L in [64, 128, 256]:
    P_va[L] = forward(X_va, Wq[L], b)[0][-1]
    P_te[L] = forward(X_te, Wq[L], b)[0][-1]

print("=" * 60)
for L in [64, 128, 256]:
    print(f"Fixed {L:3d} levels | Test acc: {np.mean(np.argmax(P_te[L],1)==y_te):.4f}")

conf_va = P_va[64].max(axis=1)
conf_te = P_te[64].max(axis=1)

def adaptive(conf, P, Te, Tm):
    tier = np.where(conf >= Te, 0, np.where(conf >= Tm, 1, 2))
    pred = np.where(tier == 0, np.argmax(P[64], 1),
           np.where(tier == 1, np.argmax(P[128], 1), np.argmax(P[256], 1)))
    return pred, np.array([64, 128, 256])[tier], tier

best = None
for Te in [0.80, 0.85, 0.90, 0.95, 0.99]:
    for Tm in [0.40, 0.50, 0.60, 0.70, 0.80]:
        if Tm >= Te:
            continue
        pred, lv, _ = adaptive(conf_va, P_va, Te, Tm)
        key = (np.mean(pred == y_va), -lv.mean())
        if best is None or key > best[0]:
            best = (key, Te, Tm)
_, Te, Tm = best

pred, lv, tier = adaptive(conf_te, P_te, Te, Tm)
print(f"Adaptive | Test acc: {np.mean(pred==y_te):.4f} | Avg levels: {lv.mean():.1f} | thr=({Te},{Tm})")
print(f"Tiers easy/med/hard = {np.sum(tier==0)}/{np.sum(tier==1)}/{np.sum(tier==2)}")
print("=" * 60)