"""
Assignment 1 – Mean Squared Error

Write a function to compute Mean Squared Error.

Parameters:
- y_true: np.ndarray, shape (n,)
- y_pred: np.ndarray, shape (n,)

Returns:
- mse: float

Function Signature:
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float
"""
import numpy as np
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    return mse
