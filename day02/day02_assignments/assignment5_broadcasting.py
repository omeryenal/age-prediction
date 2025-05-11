"""
Assignment 5 – Broadcasting and Vectorized Operations

Given:
- A 1D array of shape (3,)
- A 2D array of shape (3, 3)

Perform:
- Element-wise multiplication using broadcasting
- Row-wise addition of the 1D array to the 2D array
- Column-wise subtraction of a 1D array from the 2D array (hint: use `.T` or reshape)

Requirements:
- Do not use explicit loops
- Use broadcasting only
"""
def broadcast_ops(a,b):
    mult = b * a
    row_add = b + a
    col_sub = b - a.reshape(3,1)
    return mult,row_add,col_sub