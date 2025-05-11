# Day 05 – Multivariable Linear Regression

## Goals for Today
- Understand how to extend linear regression to multiple features
- Implement predictions using vectorized matrix operations
- Calculate MSE loss with multiple features
- Compute gradients with respect to each weight
- Perform one step of vectorized gradient descent

---

## Concept Explanation

### 1. Hypothesis with Multiple Features

Instead of a single x, we now have multiple inputs: \( x_1, x_2, ..., x_n \)

\[
\hat{y} = W_1x_1 + W_2x_2 + \cdots + W_nx_n + b
\]

This is equivalent to:

\[
\hat{y} = \mathbf{X} \cdot \mathbf{W} + b
\]

Where:
- \( \mathbf{X} \) is shape (n_samples, n_features)
- \( \mathbf{W} \) is shape (n_features,)
- \( b \) is scalar
- \( \hat{y} \) is shape (n_samples,)

---

### 2. MSE Loss Function

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

This measures the average squared difference between actual and predicted values.

---

### 3. Vectorized Gradients

\[
\frac{\partial \text{MSE}}{\partial \mathbf{W}} = -\frac{2}{n} \mathbf{X}^T (y - \hat{y})
\]
\[
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})
\]

---

### 4. Gradient Descent Step (Vectorized)

\[
\mathbf{W} := \mathbf{W} - \alpha \cdot \frac{\partial \text{MSE}}{\partial \mathbf{W}}
\]
\[
b := b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]

Where \( \alpha \) is the learning rate.

---

## Code Example

```python
import numpy as np

# Example data: 3 samples, 2 features
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([10, 20, 30])

# Initialize
W = np.zeros(X.shape[1])
b = 0.0
lr = 0.01

# Predict
y_pred = X @ W + b

# Loss
loss = np.mean((y - y_pred) ** 2)

# Gradients
dW = -2 * X.T @ (y - y_pred) / len(y)
db = -2 * np.mean(y - y_pred)

# Update
W -= lr * dW
b -= lr * db
