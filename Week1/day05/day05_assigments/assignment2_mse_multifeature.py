"""
Assignment 2 – MSE Loss for Multifeature Regression

Compute Mean Squared Error (MSE) between true and predicted values.

Parameters:
- y_true: np.ndarray, shape (n_samples,)
- y_pred: np.ndarray, shape (n_samples,)

Returns:
- loss: float

Function Signature:
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float
"""
import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_pred)
    MSE = (1/n) * sum((y_true - y_pred)**2)
    return MSE
