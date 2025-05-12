"""
Assignment 4 – 2D Contour Plot of Descent Path

Plot the 2D function f(w1, w2) = (w1 - 2)^2 + (w2 + 1)^2
and overlay the gradient descent path as a line on top.

Parameters:
- history: list of np.ndarray — positions visited during descent

Returns:
- None

Function Signature:
def plot_2d_contour_path(history: list[np.ndarray]) -> None
"""
import numpy as np
import matplotlib.pyplot as plt

def f(w1, w2):
    return (w1 - 2)**2 + (w2 + 1)**2

def gradient_descent_2d(start: np.ndarray, lr: float, steps: int) -> list[np.ndarray]:
    w = start.copy()
    history = [w.copy()]
    for _ in range(steps):
        dw1 = 2 * (w[0] - 2)
        dw2 = 2 * (w[1] + 1)
        grad = np.array([dw1, dw2])
        w = w - lr * grad
        history.append(w.copy())
    return history

def plot_2d_descent():
    # 1. Gradient descent adımları
    history = gradient_descent_2d(np.array([0.0, 0.0]), lr=0.1, steps=20)
    history = np.array(history)  # easier for plotting

    # 2. Contour plot için grid
    w1_vals = np.linspace(-1, 4, 100)
    w2_vals = np.linspace(-3, 2, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = f(W1, W2)

    # 3. Plot
    plt.figure(figsize=(8, 6))
    plt.contour(W1, W2, Z, levels=30, cmap='viridis')
    plt.plot(history[:, 0], history[:, 1], marker='o', color='red', label='Gradient Descent Path')
    plt.title("Gradient Descent on f(w1, w2) = (w1 - 2)^2 + (w2 + 1)^2")
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    plot_2d_descent()

