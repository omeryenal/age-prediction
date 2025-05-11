# Day 04 – Linear Regression from Scratch

## Goals for Today
- Understand the core logic behind linear regression
- Implement a model using only NumPy
- Manually compute gradients and update weights
- Train a simple model using gradient descent
- Visualize training progress

---

## Concept Explanation

### 1. The Linear Model (Hypothesis Function)

In univariate linear regression, we try to model the relationship between an input `x` and an output `y` as:

\[
\hat{y} = W \cdot x + b
\]

Where:
- \( \hat{y} \): predicted output
- \( W \): weight (slope)
- \( b \): bias (intercept)

### 2. Loss Function – Mean Squared Error (MSE)

To measure how well our predictions match actual data, we use:

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

The lower the MSE, the better the model is performing.

### 3. Computing Gradients

To minimize MSE using gradient descent, we calculate the partial derivatives:

\[
\frac{\partial \text{MSE}}{\partial W} = -\frac{2}{n} \sum x_i (y_i - \hat{y}_i)
\]
\[
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)
\]

These gradients point in the direction we need to adjust W and b.

### 4. Gradient Descent Update Rule

\[
W = W - \alpha \cdot \frac{\partial \text{MSE}}{\partial W}
\]
\[
b = b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]

Where \( \alpha \) is the learning rate — a small constant like 0.01 or 0.001.

---

## Code Example: One Step of Training

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])  # True model: y = 2x + 1

# Initialize parameters
W = 0.0
b = 0.0
lr = 0.01

# Prediction
y_pred = W * x + b

# Loss
loss = np.mean((y - y_pred)**2)

# Gradients
dW = -2 * np.mean(x * (y - y_pred))
db = -2 * np.mean(y - y_pred)

# Update
W -= lr * dW
b -= lr * db
