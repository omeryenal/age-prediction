"""
Assignment 4 – Reshape and Flatten

Given a 1D array with values from 0 to 15:
- Reshape it into a 4x4 matrix
- Extract the diagonal elements
- Flatten the matrix into a 1D array
- Use both `reshape`, `ravel`, and `flatten` and compare results

Requirements:
- Explore memory behavior differences between flatten and ravel
"""
import numpy as np

def reshape_and_flatten(arr):
    reshaped = arr.reshape(4,4)
    diagonal = np.diag(reshaped)
    flattened = reshaped.flatten()
    raveled = reshaped.ravel()

    return reshaped, diagonal, flattened, raveled

