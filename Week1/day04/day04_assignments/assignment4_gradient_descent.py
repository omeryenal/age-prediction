"""
Assignment 4 – One Step of Gradient Descent

Write a function `gradient_descent_step(W, b, dW, db, learning_rate)` that updates the parameters:

    W_new = W - learning_rate * dW
    b_new = b - learning_rate * db

Parameters:
- W: float
- b: float
- dW: float
- db: float
- learning_rate: float

Returns:
- W_new: float
- b_new: float

Function Signature:
def gradient_descent_step(W: float, b: float, dW: float, db: float, learning_rate: float) -> tuple[float, float]
"""
import numpy as np

def gradient_descent_step(W, b, dW, db, learning_rate):
    W_new = W - learning_rate * dW
    b_new = b - learning_rate * db
    return W_new, b_new