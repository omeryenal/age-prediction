"""
Assignment 3 – Plot Train vs Test Fit

Plot polynomial fits on both training and testing data.

Parameters:
- X_train: np.ndarray
- y_train: np.ndarray
- X_test: np.ndarray
- y_test: np.ndarray
- degree: int

Returns:
- None (shows matplotlib plot)

Function Signature:
def plot_train_test_fit(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        degree: int) -> None
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_fit(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        degree: int) -> None:

    def build_design_matrix(x, degree):
        x = x.flatten()
        return np.vstack([x**i for i in range(degree + 1)]).T

    # Birleştir tüm X değerlerini
    X_all = np.concatenate([X_train, X_test])
    X_plot = np.linspace(X_all.min() - 1, X_all.max() + 1, 300).reshape(-1, 1)
    X_plot_poly = build_design_matrix(X_plot, degree)

    # Fit (normal denklemlerle)
    X_train_poly = build_design_matrix(X_train, degree)
    theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

    # Tahmin eğrisi
    y_plot = X_plot_poly @ theta

    # Çizim
    plt.scatter(X_train, y_train, color="blue", label="Train Data")
    plt.scatter(X_test, y_test, color="green", label="Test Data")
    plt.plot(X_plot, y_plot, color="red", label=f"Polynomial Degree {degree}")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Train/Test Polynomial Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
