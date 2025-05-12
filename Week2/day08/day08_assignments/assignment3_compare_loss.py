"""
Assignment 3 – Compare Two Predictions

Given true values and two prediction arrays, compute both MSE and MAE for each prediction.

Parameters:
- y_true: np.ndarray, shape (n,)
- y_pred1: np.ndarray, shape (n,)
- y_pred2: np.ndarray, shape (n,)

Returns:
- result: dict with keys:
  {
    "mse1": float,
    "mae1": float,
    "mse2": float,
    "mae2": float
  }

Function Signature:
def compare_loss(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict
"""
import numpy as np

def compare_loss(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    mse1 = np.mean((y_true - y_pred1)**2)
    mse2 = np.mean((y_true - y_pred2)**2)

    mae1 = np.mean(abs(y_true - y_pred1))
    mae2 = np.mean(abs(y_true - y_pred2))
    result = {
        "mse1": mse1,
        "mse2": mse2,
        "mae1": mae1,
        "mae2": mae2
    }
    return result
