import numpy as np
from nn_numpy import TwoLayerNet, gradient_check, load_data, load_data_3way, train
def test_gradient_check():
    Xtr, _, ytr, _ = load_data()
    net = TwoLayerNet(64, 8, 10, seed=1)
    assert gradient_check(net, Xtr[:5], ytr[:5]) < 1e-6


def test_softmax_rows_sum_to_one():
    Xtr, _, _, _ = load_data()
    P = TwoLayerNet(64, 8, 10).forward(Xtr[:10])
    assert np.allclose(P.sum(axis=1), 1)


def test_training_reduces_loss():
    Xtr, _, ytr, _ = load_data()
    net = TwoLayerNet(64, 16, 10)
    h = train(net, Xtr, ytr, epochs=20)
    assert h[-1] < h[0]


def test_beats_chance_on_test_set():
    Xtr, Xte, ytr, yte = load_data()
    net = TwoLayerNet(64, 32, 10)
    train(net, Xtr, ytr, epochs=50)
    assert (net.predict(Xte) == yte).mean() > 0.9

def test_three_way_split():  # checks sizes and that all digits appear
    Xtr, Xval, Xte, ytr, yval, yte = load_data_3way()  # load the shared split
    assert len(Xtr) + len(Xval) + len(Xte) == 1797  # every image is in exactly one part
    assert len(Xtr) > len(Xval) and abs(len(Xval) - len(Xte)) <= 1  # train is largest, val and test are equal
    assert set(ytr) == set(range(10))  # train contains all ten digits
    a = load_data_3way()[0]  # first call, train features
    b = load_data_3way()[0]  # second call, train features
    assert np.array_equal(a, b)  # same seed gives the same split every time