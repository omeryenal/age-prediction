"""
Assignment 5 – Gradient Descent on Custom Function

Perform gradient descent using a user-defined loss and gradient function.

Parameters:
- loss_fn: callable — function f(w)
- grad_fn: callable — gradient df/dw
- start: float — initial w value
- lr: float — learning rate
- steps: int — number of iterations

Returns:
- history: list of float — values of w over time

Function Signature:
def descent_on_custom_loss(loss_fn, grad_fn, start: float, lr: float, steps: int) -> list[float]
"""

def descent_on_custom_loss(loss_fn, grad_fn, start: float, lr: float, steps: int) -> list[float]:
    w = start
    history = [w]

    for _ in range(steps):
        grad = grad_fn(w)         # gradyanı hesapla
        w = w - lr * grad         # gradient descent adımı
        history.append(w)         # w'yi sakla

    return history

