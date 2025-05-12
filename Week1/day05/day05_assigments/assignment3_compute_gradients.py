"""
Assignment 3 – Compute Vectorized Gradients

Compute gradients of MSE loss w.r.t. weights and bias.

Parameters:
- X: np.ndarray, shape (n_samples, n_features)
- y_true: np.ndarray, shape (n_samples,)
- y_pred: np.ndarray, shape (n_samples,)

Returns:
- dW: np.ndarray, shape (n_features,)
- db: float

Function Signature:
def compute_gradients(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]
"""
import numpy as np

def compute_gradients(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(X)
    dW = -2 * X.T @ (y_true - y_pred) / n
    db = -2 * np.mean(y_true - y_pred)
    return (dW, db)