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

def check(seed, Te=0.80, Tm=0.40):
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    Wq = {L: [quantize(w, L) for w in W] for L in [64, 128, 256]}
    P = {}
    for L in [64, 128, 256]:
        p, _ = forward(X_te, Wq[L], b)
        P[L] = np.argmax(p[-1], axis=1)

    p64, _ = forward(X_te, Wq[64], b)
    conf = p64[-1].max(axis=1)
    tier = np.where(conf >= Te, 0, np.where(conf >= Tm, 1, 2))
    adapt = np.where(tier == 0, P[64], np.where(tier == 1, P[128], P[256]))

    print(f"seed {seed}: "
          f"fix64!=fix128 on {np.sum(P[64] != P[128])} samples | "
          f"adapt!=fix64 on {np.sum(adapt != P[64])} | "
          f"adapt!=fix128 on {np.sum(adapt != P[128])} | "
          f"tiers easy/med/hard = {np.sum(tier==0)}/{np.sum(tier==1)}/{np.sum(tier==2)}")

for s in [0, 1, 2, 3, 4]:
    check(s)