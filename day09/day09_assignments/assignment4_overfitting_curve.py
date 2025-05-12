"""
Assignment 4 – Overfitting Curve

Evaluate train and test MSE for increasing polynomial degrees and plot the curves.

Parameters:
- X: np.ndarray
- y: np.ndarray
- degrees: list of int

Returns:
- None (displays plot)

Function Signature:
def overfitting_curve(X: np.ndarray, y: np.ndarray, degrees: list[int]) -> None
"""
import numpy as np
import matplotlib.pyplot as plt

def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def polynomial_fit_and_eval(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            degree: int) -> tuple[float, float]:
    def build_X(x, d):
        x = x.flatten()
        return np.vstack([x**i for i in range(d+1)]).T

    X_train_poly = build_X(X_train, degree)
    X_test_poly = build_X(X_test, degree)
    theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    y_train_pred = X_train_poly @ theta
    y_test_pred = X_test_poly @ theta
    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)
    return train_mse, test_mse

def overfitting_curve(X: np.ndarray, y: np.ndarray, degrees: list[int]) -> None:
    # Train/test split
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.2)

    train_errors = []
    test_errors = []

    for degree in degrees:
        train_mse, test_mse = polynomial_fit_and_eval(X_train, y_train, X_test, y_test, degree)
        train_errors.append(train_mse)
        test_errors.append(test_mse)

    # Plotting
    plt.plot(degrees, train_errors, label="Train MSE", marker='o')
    plt.plot(degrees, test_errors, label="Test MSE", marker='s')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.title("Overfitting Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
