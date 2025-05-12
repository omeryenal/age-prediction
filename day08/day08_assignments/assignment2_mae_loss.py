"""
Assignment 2 – Mean Absolute Error

Write a function to compute Mean Absolute Error.

Parameters:
- y_true: np.ndarray, shape (n,)
- y_pred: np.ndarray, shape (n,)

Returns:
- mae: float

Function Signature:
def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float
"""
import numpy as np

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mae = np.mean(abs(y_true - y_pred))
    return mae
