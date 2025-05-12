"""
Assignment 2 – Standard Deviation Calculation

Write a function to calculate the standard deviation of a 1D array.

Parameters:
- arr: np.ndarray, shape (n,)

Returns:
- std_dev: float

Function Signature:
def calculate_std(arr: np.ndarray) -> float
"""
import numpy as np
import math
def calculate_std(arr: np.ndarray) -> float:
    n = len(arr)
    mean = sum(arr) / n
    squared_diffs = [(x - mean) ** 2 for x in arr]
    variance = sum(squared_diffs) / n
    std_dev = math.sqrt(variance)
    return std_dev
    