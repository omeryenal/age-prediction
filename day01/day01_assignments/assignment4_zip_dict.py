"""
Assignment 4 – Create Dictionary from Lists

Write a function `create_dict(keys, values)` that returns a dictionary mapping elements of `keys` to elements of `values`.

Requirements:
- Raise ValueError if lists are not the same length

Example:
create_dict(["a", "b"], [1, 2]) -> {"a": 1, "b": 2}
"""
def create_dict(keys, values):
    if len(keys) != len(values):
        raise ValueError("Lists must be the same length")
    return dict(zip(keys, values))