"""
Assignment 2 – Element-wise Array Math

Create two 1D arrays of size 5 with arbitrary values.
Perform the following operations:
- Element-wise addition
- Element-wise multiplication
- Element-wise division
- Compute the square of each element

Then calculate:
- The mean and standard deviation of the resulting arrays

Requirements:
- Use NumPy built-in functions (e.g. `np.mean`, `np.std`, `np.square`)
"""


import numpy as np

def array_math_ops(a, b):
    # Element-wise operations
    addition = np.add(a, b)
    multiplication = np.multiply(a, b)
    division = np.divide(a, b)
    square = np.square(a)  # Sadece A'nın karesini alıyorsun testte

    # Statistics on addition
    mean_add = np.mean(addition)
    std_add = np.std(addition)

    return addition, multiplication, division, square, mean_add, std_add
