import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape
    w = np.zeros(n)
    b = float(0.)
    # Write code here
    for step in range (steps):
        z = X.dot(w) + b
        a = _sigmoid(z)
        error = a - y
        dw = X.T.dot(error) / m
        db = error.mean()
        w -= lr * dw
        b -= lr * db

        if np.linalg.norm(lr * dw) < 1e-6 and abs(lr * db) < 1e-6:
            break

    return w, b