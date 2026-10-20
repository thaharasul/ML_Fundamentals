import numpy as np  


class LinearRegressionGD: 
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0
        self.losses = []

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        for _ in range(self.n_iter):
            err = X @ self.w + self.b - y
            self.losses.append(float(np.mean(err ** 2)))
            self.w -= self.lr * (2 / n) * (X.T @ err)
            self.b -= self.lr * (2 / n) * err.sum()
        return self

    def predict(self, X):
        return X @ self.w + self.b