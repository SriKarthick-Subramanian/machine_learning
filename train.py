
import time
import numpy as np
from data_generation import generate_data
from model import loss_and_grad, sigmoid
from optimizers import SGD, Momentum, Adam

def train(optimizer, name, epochs=200):
    X, y = generate_data()
    w = np.zeros(X.shape[1])
    history = []
    start = time.time()
    for e in range(epochs):
        loss, grad = loss_and_grad(X, y, w)
        w = optimizer.step(w, grad)
        preds = (sigmoid(X @ w) > 0.5).astype(int)
        acc = (preds == y).mean()
        history.append((loss, acc))
    elapsed = time.time() - start
    final_loss, final_acc = history[-1]
    return name, final_loss, final_acc, elapsed

if __name__ == "__main__":
    results = []
    results.append(train(SGD(lr=0.05), "SGD"))
    results.append(train(Momentum(lr=0.05), "Momentum"))
    results.append(train(Adam(lr=0.05), "Adam"))

    for r in results:
        print(f"{r[0]} | Loss: {r[1]:.4f} | Acc: {r[2]:.4f} | Time: {r[3]:.2f}s")
