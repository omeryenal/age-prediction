"""
Assignment 5 – Outlier Impact on Loss

Demonstrate how a single outlier affects MSE more than MAE.

Returns:
- result: dict with keys:
  {
    "mse_no_outlier": float,
    "mse_with_outlier": float,
    "mae_no_outlier": float,
    "mae_with_outlier": float
  }

Function Signature:
def outlier_impact_analysis() -> dict
"""

import numpy as np

import numpy as np

def outlier_impact_analysis() -> dict:
    y_true = np.array([2, 4, 6])
    y_pred = np.array([2.1, 3.9, 6.2])

    # Add a strong outlier with wrong prediction
    y_true_outlier = np.append(y_true, 100)
    y_pred_outlier = np.append(y_pred, 5)  # deliberately wrong

    # Compute losses
    mse_no_outlier = np.mean((y_true - y_pred) ** 2)
    mae_no_outlier = np.mean(np.abs(y_true - y_pred))

    mse_with_outlier = np.mean((y_true_outlier - y_pred_outlier) ** 2)
    mae_with_outlier = np.mean(np.abs(y_true_outlier - y_pred_outlier))

    return {
        "mse_no_outlier": mse_no_outlier,
        "mse_with_outlier": mse_with_outlier,
        "mae_no_outlier": mae_no_outlier,
        "mae_with_outlier": mae_with_outlier
    }


