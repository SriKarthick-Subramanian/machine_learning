
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_and_grad(X, y, w):
    preds = sigmoid(X @ w)
    loss = -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))
    grad = X.T @ (preds - y) / X.shape[0]
    return loss, grad
