"""
Assignment 3 – 2D Gradient Descent

Perform gradient descent on the function:
f(w1, w2) = (w1 - 2)^2 + (w2 + 1)^2

Parameters:
- start: np.ndarray of shape (2,)
- lr: float — learning rate
- steps: int — number of iterations

Returns:
- history: list of np.ndarray — positions at each step

Function Signature:
def gradient_descent_2d(start: np.ndarray, lr: float, steps: int) -> list[np.ndarray]
"""
import numpy as np
def gradient_descent_2d(start: np.ndarray, lr: float, steps: int) -> list[np.ndarray]:
    w = start.copy()
    history = []

    for _ in range(steps):
        dw1 = 2 * (w[0] - 2)
        dw2 = 2 * (w[1] + 1)
        grad = np.array([dw1, dw2])
        w = w - lr * grad
        history.append(w.copy())  # sadece adım sonuçları kaydediliyor

    return history


