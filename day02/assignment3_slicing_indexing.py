"""
Assignment 3 – Indexing and Slicing

Given a 4x4 array, perform the following:
- Extract the first row
- Extract the last column
- Extract the submatrix from rows 1 to 2 and columns 1 to 3
- Reverse the entire array (hint: slicing)

Requirements:
- Use slicing and integer indexing techniques
"""
def slice_and_index(mat):
    first_row = mat[0]
    last_col = mat[: ,-1]
    submatrix =mat[1:3, 1:4]
    reversed_mat = mat[::-1, ::-1]

    return first_row, last_col, submatrix, reversed_mat