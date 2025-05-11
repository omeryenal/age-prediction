"""
Assignment 5 – Save and Load Numbers to File

Create two functions:
1. `save_numbers_to_file(numbers, filename)` – saves each number to a new line in a .txt file.
2. `load_numbers_from_file(filename)` – reads numbers from a file and returns them as a list of integers.

Requirements:
- One number per line
- Handle file read/write cleanly

Example:
save_numbers_to_file([1,2,3], "nums.txt")
load_numbers_from_file("nums.txt") -> [1, 2, 3]
"""
def save_numbers_to_file(numbers, filename):
    with open(filename, "w") as f:
        for num in numbers:
            f.write(f"{num}\n")

def load_numbers_from_file(filename):
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f]
