"""
Assignment 3 – Covariance Calculation

Write a function to calculate the covariance between two 1D arrays.

Parameters:
- x: np.ndarray, shape (n,)
- y: np.ndarray, shape (n,)

Returns:
- covariance: float

Function Signature:
def calculate_covariance(x: np.ndarray, y: np.ndarray) -> float
"""
import numpy as np

def calculate_covariance(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    covariance_sum = 0.0
    for i in range(n):
        covariance_sum += (x[i] - mean_x) * (y[i] - mean_y)

    covariance = covariance_sum / n  # Popülasyon kovaryansı
    return covariance
