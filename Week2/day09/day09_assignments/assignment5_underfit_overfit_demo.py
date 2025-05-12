"""
Assignment 5 – Underfit, Good Fit, Overfit Demo

Fit 3 models of increasing complexity and visualize the results.

Returns:
- None (shows comparison plot)

Function Signature:
def underfit_overfit_demo() -> None
"""
import numpy as np
import matplotlib.pyplot as plt

def underfit_overfit_demo() -> None:
    # 1. Veri üret
    np.random.seed(0)
    X = np.sort(np.random.rand(30, 1), axis=0)
    y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(30)

    degrees = [1, 4, 15]  # underfit, good fit, overfit

    x_plot = np.linspace(0, 1, 500).reshape(-1, 1)

    def build_X(x, d):
        x = x.flatten()
        return np.vstack([x**i for i in range(d+1)]).T

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="black", label="Data", zorder=5)

    colors = ['red', 'green', 'blue']
    labels = ['Underfit (deg=1)', 'Good Fit (deg=4)', 'Overfit (deg=15)']

    for deg, color, label in zip(degrees, colors, labels):
        X_poly = build_X(X, deg)
        X_plot_poly = build_X(x_plot, deg)
        theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
        y_plot = X_plot_poly @ theta
        plt.plot(x_plot, y_plot, color=color, label=label)

    plt.title("Underfit vs Good Fit vs Overfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
