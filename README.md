# Deep Learning Lab

Hands-on implementations built while learning deep learning.

## M4: Neural Network From Scratch (NumPy)
2-layer network (64 → 32 tanh → 10 softmax) with hand-written backprop,
trained on sklearn digits.

**Run:** `cd m4_numpy_nn && python nn_numpy.py && pytest -q`

| Metric | Result |
|---|---|
| Gradient check (relative error) | 1.6e-10 |
| Test accuracy | 97.56% |
| Test macro-F1 | 97.55% |

![loss curve](m4_numpy_nn/loss_curve.png)

**Validation:** analytic gradients match numerical gradients; 4 pytest tests.
**Limitations:** small 8×8 dataset, no regularization, single split.
