
import numpy as np
from data_generation import generate_data
from model import loss_and_grad, sigmoid
from optimizers import SGD, Momentum, Adam

def train(optimizer, epochs=20):
    X, y = generate_data()
    w = np.zeros(X.shape[1])
    history = []
    for e in range(epochs):
        loss, grad = loss_and_grad(X, y, w)
        w = optimizer.step(w, grad)
        preds = (sigmoid(X @ w) > 0.5).astype(int)
        acc = (preds == y).mean()
        history.append((loss, acc))
        print(f"Epoch {e+1}: loss={loss:.4f}, acc={acc:.4f}")
    return history

if __name__ == "__main__":
    print("SGD")
    train(SGD(lr=0.1))
    print("Momentum")
    train(Momentum(lr=0.05))
    print("Adam")
    train(Adam(lr=0.01))
