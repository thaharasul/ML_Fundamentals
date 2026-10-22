import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.X = X
        self.H = np.tanh(X @ self.W1 + self.b1)
        logits = self.H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(logits)
        self.P = e / e.sum(axis=1, keepdims=True)
        return self.P

    def loss(self, X, y):
        P = self.forward(X)
        return -np.log(P[np.arange(len(y)), y] + 1e-12).mean()

    def backward(self, y):
        n = len(y)
        dlogits = self.P.copy()
        dlogits[np.arange(n), y] -= 1     
        dlogits /= n
        dW2 = self.H.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dH = dlogits @ self.W2.T
        dZ1 = dH * (1 - self.H ** 2)       
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return [dW1, db1, dW2, db2]

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def numerical_grad(net, X, y, eps=1e-5):
    grads = []
    for p in net.params():
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps; lp = net.loss(X, y)
            p[i] = old - eps; lm = net.loss(X, y)
            p[i] = old
            g[i] = (lp - lm) / (2 * eps)
        grads.append(g)
    return grads


def gradient_check(net, X, y):
    net.loss(X, y)
    analytic = net.backward(y)
    numeric = numerical_grad(net, X, y)
    a = np.concatenate([g.ravel() for g in analytic])
    n = np.concatenate([g.ravel() for g in numeric])
    return np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n))


def load_data(seed=42):
    d = load_digits()
    X, y = d.data / 16.0, d.target
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
def load_data_3way(seed=42):  # shared split used by BOTH M4 and M5: 60% train, 20% validation, 20% test
    d = load_digits()  # 1797 images of handwritten digits, 8x8 pixels each
    X, y = d.data / 16.0, d.target  # pixels range 0-16, so dividing by 16 scales them to 0-1
    Xtr, Xtmp, ytr, ytmp = train_test_split(  # first split: 60% train, 40% held out
        X, y, test_size=0.4, stratify=y, random_state=seed  # stratify keeps each digit's share equal
    )
    Xval, Xte, yval, yte = train_test_split(  # second split: halve the held-out part
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed  # 20% validation, 20% test
    )
    return Xtr, Xval, Xte, ytr, yval, yte  # hand back all six pieces

def train(net, Xtr, ytr, epochs=100, lr=0.5, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        idx = rng.permutation(len(ytr))
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            net.forward(Xtr[b])
            grads = net.backward(ytr[b])
            for p, g in zip(net.params(), grads):
                p -= lr * g
        history.append(net.loss(Xtr, ytr))
    return history


if __name__ == "__main__":
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xtr, Xte, ytr, yte = load_data()
    net = TwoLayerNet(64, 32, 10)
    err = gradient_check(net, Xtr[:5], ytr[:5])
    print(f"gradient check relative error: {err:.2e}")
    hist = train(net, Xtr, ytr)
    pred = net.predict(Xte)
    res = {"grad_check_rel_error": float(err),
           "final_train_loss": float(hist[-1]),
           "test_accuracy": float(accuracy_score(yte, pred)),
           "test_macro_f1": float(f1_score(yte, pred, average="macro"))}
    print(res)
    json.dump(res, open("results_m4.json", "w"), indent=2)
    plt.plot(hist); plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("M4 NumPy NN")
    plt.savefig("loss_curve.png", dpi=120)