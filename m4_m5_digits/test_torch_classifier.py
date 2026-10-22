import numpy as np

from nn_numpy import load_data_3way
from torch_classifier import make_model, train, save_checkpoint, load_checkpoint
import torch

def test_output_shape():
    out = make_model()(torch.zeros(5, 64))
    assert tuple(out.shape) == (5, 10)


def test_training_reduces_loss():
    Xtr, Xval, _, ytr, yval, _ = load_data_3way()
    h = train(make_model(), Xtr, ytr, Xval, yval, epochs=10)
    assert h["train_loss"][-1] < h["train_loss"][0]


def test_same_seed_same_result():
    Xtr, Xval, _, ytr, yval, _ = load_data_3way()
    a = train(make_model(), Xtr, ytr, Xval, yval, epochs=3)["train_loss"][-1]
    b = train(make_model(), Xtr, ytr, Xval, yval, epochs=3)["train_loss"][-1]
    assert np.isclose(a, b)


def test_beats_chance_on_validation():
    Xtr, Xval, _, ytr, yval, _ = load_data_3way()
    h = train(make_model(), Xtr, ytr, Xval, yval, epochs=30)
    assert h["val_acc"][-1] > 0.9