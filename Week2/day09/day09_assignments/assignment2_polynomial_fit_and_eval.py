"""
Assignment 2 – Polynomial Fit and Evaluate MSE

Fit a polynomial model and return train and test MSE.

Parameters:
- X_train: np.ndarray
- y_train: np.ndarray
- X_test: np.ndarray
- y_test: np.ndarray
- degree: int

Returns:
- train_mse: float
- test_mse: float

Function Signature:
def polynomial_fit_and_eval(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            degree: int) -> tuple[float, float]
"""
import numpy as np
def polynomial_fit_and_eval(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            degree: int) -> tuple[float, float]:

    # Design matrix oluştur (X^0, X^1, ..., X^d)
    def build_design_matrix(x, degree):
        x = x.flatten()
        return np.vstack([x**i for i in range(degree + 1)]).T

    X_train_poly = build_design_matrix(X_train, degree)
    X_test_poly = build_design_matrix(X_test, degree)

    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

    # Tahminler
    y_train_pred = X_train_poly @ theta
    y_test_pred = X_test_poly @ theta

    # Hatalar
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)

    return train_mse, test_mse