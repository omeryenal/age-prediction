"""
Assignment 1 – Prime Number Checker

Write a function `is_prime(n)` that takes an integer `n` and returns True if it is a prime number, False otherwise.

Requirements:
- Must work for all integers n >= 0
- Should be efficient for numbers up to at least 10,000

Example:
is_prime(7) -> True
is_prime(10) -> False
"""

def is_prime(n):
    if n == 0:
        return False
    elif n == 1:
        return False
    elif n == 2:
        return True
    for i in range(2,n):
        for j in range(2,i):
            if n % j == 0 and n != j:
                return False
    return True

print(is_prime(7))
print(is_prime(10))
print(is_prime(6))
print(is_prime(14))