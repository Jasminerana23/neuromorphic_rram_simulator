import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from crossbar_array import CrossbarArray

G_MIN = 1 / 100e3   # 100 kOhm
G_MAX = 1 / 1e3     # 1 kOhm
LOAD = 1000.0       # CrossbarArray.forward() multiplies output by 1000

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

def reference_weights(W, L):
    Wmax = np.abs(W).max()
    return Wmax * np.round(W / Wmax * (L - 1)) / (L - 1)

def map_layer(W, L):
    # Differential pair conductances (G+ and G-), each on L levels in [G_MIN, G_MAX]
    Wmax = np.abs(W).max()
    kp = np.round(np.maximum(W, 0) / Wmax * (L - 1))
    kn = np.round(np.maximum(-W, 0) / Wmax * (L - 1))
    Gp = G_MIN + (G_MAX - G_MIN) * kp / (L - 1)
    Gn = G_MIN + (G_MAX - G_MIN) * kn / (L - 1)
    return Gp, Gn, Wmax

def build_pair(Gp, Gn):
    rows, cols = Gp.shape
    cp = CrossbarArray(rows=rows, cols=cols)
    cn = CrossbarArray(rows=rows, cols=cols)
    cp.set_weights(1.0 / Gp)    # set resistance = 1/G
    cn.set_weights(1.0 / Gn)
    return cp, cn

def crossbar_layer(a, cp, cn, Wmax):
    # Read the resistances stored inside the CrossbarArray objects
    Gp = 1.0 / cp.get_weights()
    Gn = 1.0 / cn.get_weights()
    I = a @ (Gp - Gn)
    return I * Wmax / (G_MAX - G_MIN)

def check_class_forward(cp, cn, Wmax, x):
    # Compare CrossbarArray.forward() (device-by-device loop) with the vectorized math
    v_pos = np.array(cp.forward(list(x))) / LOAD
    v_neg = np.array(cn.forward(list(x))) / LOAD
    class_out = (v_pos - v_neg) * Wmax / (G_MAX - G_MIN)
    vec_out = crossbar_layer(x[None, :], cp, cn, Wmax)[0]
    return np.max(np.abs(class_out - vec_out))

def run_numpy(X, Wq, b):
    a = X
    for i in range(3):
        z = a @ Wq[i] + b[i]
        a = np.maximum(0, z) if i < 2 else softmax(z)
    return a

def run_xbar(X, xbars, b):
    a = X
    for i, (cp, cn, Wm) in enumerate(xbars):
        z = crossbar_layer(a, cp, cn, Wm) + b[i]
        a = np.maximum(0, z) if i < 2 else softmax(z)
    return a

def run_seed(seed, L_list=(64, 128, 256)):
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

    sizes = [64, 128, 64, 10]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i]) for i in range(3)]
    b = [np.zeros(sizes[i+1]) for i in range(3)]
    train(X_tr, y_tr, W, b)

    out = {}
    for L in L_list:
        Wq = [reference_weights(w, L) for w in W]
        p_np = run_numpy(X_te, Wq, b)

        xbars = []
        for w in W:
            Gp, Gn, Wm = map_layer(w, L)
            cp, cn = build_pair(Gp, Gn)
            xbars.append((cp, cn, Wm))
        p_xb = run_xbar(X_te, xbars, b)

        acc_np = np.mean(np.argmax(p_np, axis=1) == y_te)
        acc_xb = np.mean(np.argmax(p_xb, axis=1) == y_te)
        max_diff = np.max(np.abs(p_np - p_xb))
        out[L] = (acc_np, acc_xb, max_diff)

        if seed == 0 and L == 64:
            cdiff = check_class_forward(xbars[0][0], xbars[0][1], xbars[0][2], X_te[0])
            print(f"  [class check] CrossbarArray.forward() vs vectorized, layer 1: max|diff|={cdiff:.2e}")
    return out

all_res = {L: [] for L in (64, 128, 256)}
for s in range(5):
    res = run_seed(s)
    for L in res:
        acc_np, acc_xb, md = res[L]
        all_res[L].append((acc_np, acc_xb))
        print(f"seed {s} L={L:3d}: numpy={acc_np:.4f} pipeline={acc_xb:.4f} max|diff|={md:.2e}")

print("=" * 70)
print("MEAN ± STD over 5 seeds (symmetric quantizer, test set)")
for L in (64, 128, 256):
    arr = np.array(all_res[L])
    print(f"  L={L:3d}  numpy: {arr[:,0].mean():.4f} ± {arr[:,0].std():.4f}"
          f"   pipeline: {arr[:,1].mean():.4f} ± {arr[:,1].std():.4f}")
print("=" * 70)