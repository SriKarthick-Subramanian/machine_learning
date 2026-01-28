
import numpy as np

def generate_data(n_samples=100_000, n_features=500, sparsity=0.9, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 1 - sparsity, size=(n_samples, n_features)).astype(float)
    true_w = rng.normal(0, 1, size=n_features)
    logits = X @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return X, y
