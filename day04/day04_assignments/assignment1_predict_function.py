"""
Assignment 1 – Predict Function (Linear Hypothesis)

Write a function `predict(x, W, b)` that computes predictions using a linear model:

    ŷ = W * x + b

Parameters:
- x: np.ndarray of shape (n,)
- W: scalar (float) – weight
- b: scalar (float) – bias

Returns:
- y_pred: np.ndarray of shape (n,)

Function Signature:
def predict(x: np.ndarray, W: float, b: float) -> np.ndarray
"""
import numpy as np

def predict(x, W, b):
    y_hat = W * x + b
    return y_hat
