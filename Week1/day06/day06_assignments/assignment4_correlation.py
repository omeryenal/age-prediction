"""
Assignment 4 – Pearson Correlation Coefficient

Write a function to calculate the Pearson correlation between two 1D arrays.

Parameters:
- x: np.ndarray, shape (n,)
- y: np.ndarray, shape (n,)

Returns:
- correlation: float

Function Signature:
def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float
"""
import numpy as np

def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:

    cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    corr_xy = cov_xy / (np.std(x) * np.std(y))
    return corr_xy
