import numpy as np  # array maths for the predictions
import torch  # PyTorch: tensors and automatic gradients
from torch import nn  # building blocks: layers and loss functions
from torch.utils.data import DataLoader, TensorDataset  # feeds data to the model in mini-batches
from sklearn.metrics import accuracy_score, f1_score  # same metrics as M4


def make_model(n_in=64, n_hidden=32, n_out=10, seed=0):  # same shape as the M4 network
    torch.manual_seed(seed)  # fixed seed so the starting weights are the same every run
    return nn.Sequential(  # layers applied one after another
        nn.Linear(n_in, n_hidden),  # 64 pixels -> 32 hidden values
        nn.Tanh(),  # same activation as M4
        nn.Linear(n_hidden, n_out),  # 32 hidden values -> 10 digit scores (logits)
    )  # no softmax here: CrossEntropyLoss applies it internally


def make_loader(X, y, batch=64, shuffle=False, seed=0):  # wraps arrays into a mini-batch loader
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))  # pixels as floats, labels as integers
    g = torch.Generator()  # own random generator, so shuffling is repeatable
    g.manual_seed(seed)  # fix its seed
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g)  # yields (images, labels) batches


def evaluate(model, X, y):  # loss and accuracy on a whole dataset, without training
    model.eval()  # switch to evaluation mode
    loss_fn = nn.CrossEntropyLoss()  # softmax + cross-entropy in one
    with torch.no_grad():  # don't track gradients: saves memory and time
        logits = model(torch.tensor(X, dtype=torch.float32))  # raw scores for every image
        loss = loss_fn(logits, torch.tensor(y, dtype=torch.long)).item()  # average loss as a plain number
        pred = logits.argmax(dim=1).numpy()  # the digit with the highest score
    return loss, float(accuracy_score(y, pred))  # loss and accuracy


def predict(model, X):  # predicted digits for new images
    model.eval()  # evaluation mode
    with torch.no_grad():  # no gradients needed
        return model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()  # highest-scoring digit per image


def macro_f1(model, X, y):  # macro-F1, same metric as M4
    return float(f1_score(y, predict(model, X), average="macro"))  # average F1 over the ten digits


def train(model, Xtr, ytr, Xval, yval, epochs=100, lr=0.5, batch=64, seed=0):  # explicit training loop, no Trainer
    opt = torch.optim.SGD(model.parameters(), lr=lr)  # plain gradient descent, like M4
    loss_fn = nn.CrossEntropyLoss()  # the loss we minimise
    loader = make_loader(Xtr, ytr, batch=batch, shuffle=True, seed=seed)  # shuffled mini-batches of TRAIN data
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}  # record progress every epoch
    for _ in range(epochs):  # one pass over the training data per epoch
        model.train()  # training mode
        for xb, yb in loader:  # one mini-batch at a time
            opt.zero_grad()  # clear gradients left from the previous batch
            loss = loss_fn(model(xb), yb)  # forward pass and loss
            loss.backward()  # autograd computes every gradient (our hand-written backprop in M4)
            opt.step()  # update weights: w = w - lr * gradient
        tr_loss, _ = evaluate(model, Xtr, ytr)  # loss on all train data, measured like M4
        va_loss, va_acc = evaluate(model, Xval, yval)  # loss and accuracy on VALIDATION data
        hist["train_loss"].append(tr_loss)  # save train loss
        hist["val_loss"].append(va_loss)  # save validation loss
        hist["val_acc"].append(va_acc)  # save validation accuracy
    return hist  # history for the loss curvesdef save_checkpoint(model, path):  # save the learned weights to a file
    torch.save(model.state_dict(), path)  # state_dict = all weights and biases

def save_checkpoint(model, path):  # save the learned weights to a file
    torch.save(model.state_dict(), path)  # state_dict = all weights and biases


def load_checkpoint(path, **kwargs):  # rebuild the model and fill it with saved weights
    model = make_model(**kwargs)  # fresh model with the same shape
    model.load_state_dict(torch.load(path, weights_only=True))  # copy saved weights in
    model.eval()  # evaluation mode
    return model  # the restored model