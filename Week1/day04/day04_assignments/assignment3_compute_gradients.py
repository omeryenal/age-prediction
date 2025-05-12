"""
Assignment 3 – Compute Gradients for Linear Regression

Write a function `compute_gradients(x, y_true, y_pred)` that calculates:

    dW = -2 * mean(x * (y_true - y_pred))
    db = -2 * mean(y_true - y_pred)

Parameters:
- x: np.ndarray of shape (n,)
- y_true: np.ndarray of shape (n,)
- y_pred: np.ndarray of shape (n,)

Returns:
- dW: float
- db: float

Function Signature:
def compute_gradients(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]
"""
import numpy as np
def compute_gradients(x,y_true,y_pred):
    dW = -2 * np.mean(x * (y_true - y_pred))
    db = -2 * np.mean(y_true - y_pred)
    return dW, db