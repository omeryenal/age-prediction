"""
Assignment 1 – 1D Gradient Descent

Implement gradient descent for f(w) = (w - 3)^2.

Parameters:
- start: float — initial value of w
- lr: float — learning rate
- steps: int — number of iterations

Returns:
- history: list of float — values of w at each step

Function Signature:
def gradient_descent_1d(start: float, lr: float, steps: int) -> list[float]
"""

def gradient_descent_1d(start: float, lr: float, steps: int) -> list[float]:
    history = []
    w = start
    for _ in range(steps):
        grad = 2* w - 6
        w = w - lr * grad
        history.append(w)
    return history
