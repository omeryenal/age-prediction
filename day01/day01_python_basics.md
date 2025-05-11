# Day 1 – Python Basics for Machine Learning

## Goals for Today
- Refresh core Python syntax
- Practice data structures and control flow
- Get comfortable writing clean, modular code
- Prepare for numerical operations with NumPy

---

## Topics Covered
- Variables and data types (`int`, `float`, `str`, `bool`)
- Data structures: `list`, `tuple`, `dict`, `set`
- Conditional statements: `if`, `elif`, `else`
- Loops: `for`, `while`
- Functions and scope
- Basic file I/O (`open()`, `read()`, `write()`)

---

## Key Learnings
- Python's dynamic typing makes it fast to prototype ML ideas.
- Functions make code more reusable and testable.
- Lists and dictionaries are essential for data preprocessing.

---

## Example Code

### 1. Basic Loop and Function
```python
def square_numbers(numbers):
    result = []
    for n in numbers:
        result.append(n ** 2)
    return result

nums = [1, 2, 3, 4, 5]
print(square_numbers(nums))  # Output: [1, 4, 9, 16, 25]
