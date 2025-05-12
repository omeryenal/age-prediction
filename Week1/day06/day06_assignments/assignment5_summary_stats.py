"""
Assignment 5 – Summary Statistics per Feature

Write a function to compute mean, variance, and std for each column (feature) in a 2D array.

Parameters:
- X: np.ndarray, shape (n_samples, n_features)

Returns:
- stats: dict[str, list[float]]
  Keys: "mean", "variance", "std"

Function Signature:
def summarize_dataset(X: np.ndarray) -> dict
"""
import numpy as np

def summarize_dataset(X: np.ndarray) -> dict:
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    stds = np.std(X, axis=0)

    return {
        "mean": means.tolist(),
        "variance": variances.tolist(),
        "std": stds.tolist()
    }
