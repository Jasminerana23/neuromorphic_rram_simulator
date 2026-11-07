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
        Gp = np.clip(Gp * (1 + sigma * rng.standard_normal(Gp.shape)), G_MIN, G_MAX)
        Gn = np.clip(Gn * (1 + sigma * rng.standard_normal(Gn.shape)), G_MIN, G_MAX)
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

def adaptive(conf, P, Te, Tm):
    tier = np.where(conf >= Te, 0, np.where(conf >= Tm, 1, 2))
    pred = np.where(tier == 0, np.argmax(P[64], 1),
           np.where(tier == 1, np.argmax(P[128], 1), np.argmax(P[256], 1)))
    return pred, np.array([64, 128, 256])[tier]

def run_seed(seed, sigma):
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    P_va, P_te = {}, {}
    for L in [64, 128, 256]:
        xb = build_xbars(W, L, sigma, rng)
        P_va[L] = run_xbar(X_va, xb, b)
        P_te[L] = run_xbar(X_te, xb, b)

    conf_va = P_va[64].max(axis=1)
    conf_te = P_te[64].max(axis=1)

    # Thresholds tuned on VALIDATION (noisy crossbar), same grid as before
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
    return fixed[64], fixed[128], fixed[256], np.mean(pred == y_te), lv.mean()

SIGMAS = [0.0, 0.05, 0.10]
print("=" * 90)
print("ADAPTIVE UNDER READ NOISE: mean ± std over 5 seeds (test set)")
print(f"{'sigma':>7}  {'Fixed 256':>16}  {'Adaptive':>16}  {'Avg levels':>12}")
for sigma in SIGMAS:
    rows = [run_seed(s, sigma) for s in range(5)]
    R = np.array(rows)
    print(f"{sigma*100:6.1f}%  "
          f"{R[:,2].mean():.4f} ± {R[:,2].std():.4f}  "
          f"{R[:,3].mean():.4f} ± {R[:,3].std():.4f}  "
          f"{R[:,4].mean():6.1f} ± {R[:,4].std():.1f}")
print("=" * 90)