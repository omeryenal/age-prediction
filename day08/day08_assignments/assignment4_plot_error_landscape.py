"""
Assignment 4 – Plot Error Landscape

For a fixed y_true, plot MSE and MAE as a function of predicted value.

Parameters:
- y_true: np.ndarray, shape (n,) – e.g., [2, 4, 6]

Returns:
- None (displays a matplotlib plot)

Function Signature:
def plot_error_landscape(y_true: np.ndarray) -> None
"""
import matplotlib.pyplot as plt
import numpy as np
def plot_error_landscape(y_true: np.ndarray) -> None:
    y_preds = np.linspace(-2,10,500)
    mse_list = []
    mae_list = []

    for y_pred in y_preds:
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        mse_list.append(mse)
        mae_list.append(mae)

    plt.plot(y_preds, mse_list, label="MSE")
    plt.plot(y_preds, mae_list, label="MAE")
    plt.xlabel("Predicted Value (y_pred)")
    plt.ylabel("Error")
    plt.title("Error Landscape: MSE vs MAE")
    plt.legend()
    plt.grid(True)
    plt.show()
