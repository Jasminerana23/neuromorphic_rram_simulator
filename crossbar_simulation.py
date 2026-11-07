import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

G_MIN = 1 / 100e3   # 100 kΩ -> 1e-5 S
G_MAX = 1 / 1e3     # 1 kΩ   -> 1e-3 S

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

def run_network(X, layer_fns, b):
    a = X
    for i, f in enumerate(layer_fns):
        z = f(a) + b[i]
        a = np.maximum(0, z) if i < 2 else softmax(z)
    return a

def reference_weights(W, L):
    Wmax = np.abs(W).max()
    return Wmax * np.round(W / Wmax * (L - 1)) / (L - 1)

def map_to_conductance(W, L):
    Wmax = np.abs(W).max()
    k_pos = np.round(np.maximum(W, 0) / Wmax * (L - 1))
    k_neg = np.round(np.maximum(-W, 0) / Wmax * (L - 1))
    G_pos = G_MIN + (G_MAX - G_MIN) * k_pos / (L - 1)
    G_neg = G_MIN + (G_MAX - G_MIN) * k_neg / (L - 1)
    return G_pos, G_neg, Wmax

def crossbar_mvm(a, G_pos, G_neg, Wmax):
    I = a @ (G_pos - G_neg)
    return I * Wmax / (G_MAX - G_MIN)

def run_seed(seed, L_list=(64, 128, 256)):
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    results = {}
    for L in L_list:
        Wref = [reference_weights(w, L) for w in W]
        ref_fns = [(lambda a, Wq=Wq: a @ Wq) for Wq in Wref]

        maps = [map_to_conductance(w, L) for w in W]
        xbar_fns = [(lambda a, gp=gp, gn=gn, wm=wm: crossbar_mvm(a, gp, gn, wm))
                    for gp, gn, wm in maps]

        p_ref = run_network(X_te, ref_fns, b)
        p_xbar = run_network(X_te, xbar_fns, b)

        acc_ref = np.mean(np.argmax(p_ref, axis=1) == y_te)
        acc_xbar = np.mean(np.argmax(p_xbar, axis=1) == y_te)
        max_diff = np.max(np.abs(p_ref - p_xbar))
        results[L] = (acc_ref, acc_xbar, max_diff)
    return results

all_res = {L: [] for L in (64, 128, 256)}
for s in range(5):
    res = run_seed(s)
    for L in res:
        acc_ref, acc_xbar, max_diff = res[L]
        all_res[L].append((acc_ref, acc_xbar))
        print(f"seed {s} L={L:3d}: numpy={acc_ref:.4f} crossbar={acc_xbar:.4f} max|diff|={max_diff:.2e}")

print("=" * 70)
print("MEAN ± STD over 5 seeds (test set)")
for L in (64, 128, 256):
    arr = np.array(all_res[L])
    print(f"  L={L:3d}  numpy: {arr[:,0].mean():.4f} ± {arr[:,0].std():.4f}"
          f"   crossbar: {arr[:,1].mean():.4f} ± {arr[:,1].std():.4f}")
print("=" * 70)