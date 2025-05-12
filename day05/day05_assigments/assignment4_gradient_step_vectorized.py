"""
Assignment 4 – One Step of Vectorized Gradient Descent

Update weights and bias using computed gradients.

Parameters:
- W: np.ndarray, shape (n_features,)
- b: float
- dW: np.ndarray, shape (n_features,)
- db: float
- learning_rate: float

Returns:
- W_new: np.ndarray, shape (n_features,)
- b_new: float

Function Signature:
def gradient_step(W: np.ndarray, b: float, dW: np.ndarray, db: float, learning_rate: float) -> tuple[np.ndarray, float]
"""
import numpy as np
def gradient_step(W, b, dW, db, learning_rate):
    W_new = W - learning_rate * dW
    b_new = b - learning_rate * db
    return W_new, b_new
