"""
Assignment 1 – Array Creation

Create the following arrays using NumPy:
1. A 1D array with values from 0 to 9
2. A 2D array of shape (3, 3) filled with ones
3. A 3x3 identity matrix
4. A linearly spaced array between 0 and 1 with 5 elements

Requirements:
- Use `np.arange`, `np.ones`, `np.eye`, and `np.linspace` where appropriate.
"""
import numpy as np
def create_0_to_9():
    OneD_array = np.arange(10)
    return OneD_array

def create_3x3_ones():
    twoD_array = np.ones([3,3])
    return twoD_array

def create_identity():
    identity_array = np.eye(3,3)
    return identity_array

def create_linspace():
    linspace_array = np.linspace(0,1,5)
    return linspace_array

