import numpy as np
from sklearn.linear_model import LinearRegression

from src.linreg import LinearRegressionGD


def make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + rng.normal(scale=0.01, size=200)
    return X, y


def test_loss_decreases():
    X, y = make_data()
    m = LinearRegressionGD(lr=0.1, n_iter=200).fit(X, y)
    assert m.losses[-1] < m.losses[0]


def test_matches_sklearn():
    X, y = make_data()
    m = LinearRegressionGD(lr=0.1, n_iter=2000).fit(X, y)
    ref = LinearRegression().fit(X, y)
    assert np.allclose(m.w, ref.coef_, atol=1e-2)
    assert abs(m.b - ref.intercept_) < 1e-2