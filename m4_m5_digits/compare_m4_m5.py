import matplotlib  # plotting library
matplotlib.use("Agg")  # draw to a file, without opening a window
import matplotlib.pyplot as plt  # plotting functions
import numpy as np  # array maths
import pandas as pd  # builds the results table
from sklearn.metrics import accuracy_score, f1_score  # same metrics for both models

from nn_numpy import TwoLayerNet, load_data_3way, train as train_numpy  # M4 model and its training loop
from torch_classifier import make_model, train as train_torch, predict, save_checkpoint, load_checkpoint  # M5 pieces


def scores(name, ytrue_val, pred_val, ytrue_te, pred_te):  # one row of the results table
    return {  # collect the numbers
        "model": name,  # which model this row describes
        "val_accuracy": accuracy_score(ytrue_val, pred_val),  # accuracy on validation data
        "test_accuracy": accuracy_score(ytrue_te, pred_te),  # accuracy on the untouched test data
        "test_macro_f1": f1_score(ytrue_te, pred_te, average="macro"),  # macro-F1 on test data
    }


def main():  # run both models on identical data
    Xtr, Xval, Xte, ytr, yval, yte = load_data_3way()  # the shared 60/20/20 split

    net = TwoLayerNet(64, 32, 10)  # M4: NumPy network
    hist_np = train_numpy(net, Xtr, ytr, epochs=100, lr=0.5, batch=64)  # same settings as M5; returns train loss per epoch

    model = make_model()  # M5: PyTorch network with the same shape
    hist_t = train_torch(model, Xtr, ytr, Xval, yval, epochs=100, lr=0.5, batch=64)  # returns a dictionary of curves

    rows = [  # one row per model
        scores("M4 NumPy", yval, net.predict(Xval), yte, net.predict(Xte)),  # NumPy results
        scores("M5 PyTorch", yval, predict(model, Xval), yte, predict(model, Xte)),  # PyTorch results
    ]
    table = pd.DataFrame(rows).round(4)  # tidy numbers
    print(table.to_string(index=False))  # show the comparison
    table.to_csv("results_m4_vs_m5.csv", index=False)  # save as evidence

    save_checkpoint(model, "m5_checkpoint.pt")  # save the PyTorch weights
    restored = load_checkpoint("m5_checkpoint.pt")  # reload them into a new model
    same = np.array_equal(predict(model, Xte), predict(restored, Xte))  # do both predict identically?
    print("Checkpoint reload gives identical predictions:", same)  # should print True

    plt.plot(hist_np, label="M4 NumPy")  # NumPy train loss per epoch
    plt.plot(hist_t["train_loss"], label="M5 PyTorch")  # PyTorch train loss per epoch
    plt.xlabel("Epoch")  # label x-axis
    plt.ylabel("Train loss")  # label y-axis
    plt.title("Training loss: NumPy vs PyTorch")  # chart title
    plt.legend()  # show which line is which
    plt.savefig("loss_compare.png", dpi=150)  # save the picture
    print("Saved loss_compare.png and results_m4_vs_m5.csv")  # confirmation


if __name__ == "__main__":  # run only when executed directly
    main()  # start the comparison