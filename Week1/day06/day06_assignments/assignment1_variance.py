"""
Assignment 1 – Variance Calculation

Write a function to calculate the variance of a 1D array.

Parameters:
- arr: np.ndarray, shape (n,)

Returns:
- variance: float

Function Signature:
def calculate_variance(arr: np.ndarray) -> float
"""
import numpy as np

def calculate_variance(arr: list[float]) -> float:
    n = len(arr)
    mean = sum(arr) / n
    squared_diffs = [(x - mean) ** 2 for x in arr]
    variance = sum(squared_diffs) / n
    return variance
