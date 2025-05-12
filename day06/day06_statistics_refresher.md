Day 06 – Statistics Refresher: Variance, Covariance, Correlation

Goals for Today

Refresh core statistical concepts relevant to ML

Understand variance and standard deviation

Compute covariance to analyze joint variability

Use correlation to understand strength of relationships

Apply NumPy for efficient statistical computation

Concept Explanation

1. Variance and Standard Deviation

Variance measures how much values in a dataset deviate from the mean.

variance = (1/n) * Σ (xᵢ - mean)²
std_dev = √(variance)

Where n is the number of elements.

2. Covariance

Covariance measures how two variables vary together:

cov(X, Y) = (1/n) * Σ (xᵢ - mean_x) * (yᵢ - mean_y)

If cov > 0: variables increase together

If cov < 0: one increases, other decreases

If cov ~ 0: no linear relationship

3. Correlation Coefficient (Pearson)

Correlation scales covariance into a range between -1 and 1:

corr(X, Y) = cov(X, Y) / (std(X) * std(Y))

1.0: perfect positive linear correlation

-1.0: perfect negative correlation

0: no correlation

Code Example: Covariance and Correlation

import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Variance
var_x = np.var(x)

# Covariance
cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))

# Correlation
corr_xy = cov_xy / (np.std(x) * np.std(y))

print("Variance:", var_x)
print("Covariance:", cov_xy)
print("Correlation:", corr_xy)

Assignments

All implementations go in day06_assignments/:

assignment1_variance.py→ def calculate_variance(arr: np.ndarray) -> float

assignment2_std.py→ def calculate_std(arr: np.ndarray) -> float

assignment3_covariance.py→ def calculate_covariance(x: np.ndarray, y: np.ndarray) -> float

assignment4_correlation.py→ def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float

assignment5_summary_stats.py→ def summarize_dataset(X: np.ndarray) -> dict(Return: mean, variance, std for each feature)

Reflection

Today I revisited statistics from a machine learning lens. Covariance and correlation help diagnose feature relationships before building models. Variance and std are also central to normalization and optimization techniques.

Next Up: Day 07 – Gradient Descent Theory + Practice

