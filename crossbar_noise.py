import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from crossbar_array import CrossbarArray

G_MIN = 1 / 100e3
G_MAX = 1 / 1e3
LOAD = 1000.0

X, y = load_digits(return_X_y=True)
X = X / 16.0

def onehot(labels):
    Y = np.zeros((len(labels), 10))
    Y[np.arange(len(labels)), labels] = 1
    return Y

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward_train(X, W, b):
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
            acts, zs = forward_train(xb, W, b)
            n = len(bi)
            delta = (acts[-1] - yb) / n
            for i in reversed(range(3)):
                dW = acts[i].T @ delta
                db = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ W[i].T) * (zs[i-1] > 0)
                W[i] -= lr * dW
                b[i] -= lr * db

def map_layer(W, L):
    Wmax = np.abs(W).max()
    kp = np.round(np.maximum(W, 0) / Wmax * (L - 1))
    kn = np.round(np.maximum(-W, 0) / Wmax * (L - 1))
    Gp = G_MIN + (G_MAX - G_MIN) * kp / (L - 1)
    Gn = G_MIN + (G_MAX - G_MIN) * kn / (L - 1)
    return Gp, Gn, Wmax

def build_xbars(W, L, sigma, rng):
    xbars = []
    for w in W:
        Gp, Gn, Wm = map_layer(w, L)
        # Multiplicative read noise on each conductance: G * (1 + sigma * N(0,1))
        Gp = Gp * (1 + sigma * rng.standard_normal(Gp.shape))
        Gn = Gn * (1 + sigma * rng.standard_normal(Gn.shape))
        Gp = np.clip(Gp, G_MIN, G_MAX)
        Gn = np.clip(Gn, G_MIN, G_MAX)
        rows, cols = Gp.shape
        cp = CrossbarArray(rows=rows, cols=cols)
        cn = CrossbarArray(rows=rows, cols=cols)
        cp.set_weights(1.0 / Gp)
        cn.set_weights(1.0 / Gn)
        xbars.append((cp, cn, Wm))
    return xbars

def crossbar_layer(a, cp, cn, Wm):
    Gp = 1.0 / cp.get_weights()
    Gn = 1.0 / cn.get_weights()
    return (a @ (Gp - Gn)) * Wm / (G_MAX - G_MIN)

def run_xbar(X, xbars, b):
    a = X
    for i, (cp, cn, Wm) in enumerate(xbars):
        z = crossbar_layer(a, cp, cn, Wm) + b[i]
        a = np.maximum(0, z) if i < 2 else softmax(z)
    return a

def run_seed(seed, sigma, L_list=(64, 128, 256)):
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    acc = {}
    for L in L_list:
        xb = build_xbars(W, L, sigma, rng)
        p = run_xbar(X_te, xb, b)
        acc[L] = np.mean(np.argmax(p, axis=1) == y_te)
    return acc

SIGMAS = [0.0, 0.02, 0.05, 0.10]
results = {}
for sigma in SIGMAS:
    results[sigma] = {L: [] for L in (64, 128, 256)}
    for seed in range(5):
        acc = run_seed(seed, sigma)
        for L in acc:
            results[sigma][L].append(acc[L])

print("=" * 70)
print("READ NOISE SWEEP: mean ± std accuracy over 5 seeds (test set)")
print(f"{'sigma':>7}  {'Fixed 64':>16}  {'Fixed 128':>16}  {'Fixed 256':>16}")
for sigma in SIGMAS:
    row = []
    for L in (64, 128, 256):
        a = np.array(results[sigma][L])
        row.append(f"{a.mean():.4f} ± {a.std():.4f}")
    print(f"{sigma*100:6.1f}%  {row[0]:>16}  {row[1]:>16}  {row[2]:>16}")
print("=" * 70)