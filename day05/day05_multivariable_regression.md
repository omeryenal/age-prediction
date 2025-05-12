Day 05 – Multivariable Linear Regression

Goals for Today

Extend linear regression to multiple input features

Use vectorized matrix operations for prediction

Calculate MSE loss in multivariable context

Compute gradients with respect to each weight

Perform one step of vectorized gradient descent

Concept Explanation

1. Hypothesis with Multiple Features

Given multiple inputs (x₁, x₂, ..., xₙ):

ŷ = W₁·x₁ + W₂·x₂ + ... + Wₙ·xₙ + b

Or in matrix form:

ŷ = X @ W + b

Where:

X is shape (n_samples, n_features)

W is shape (n_features,)

b is scalar

ŷ is shape (n_samples,)

2. MSE Loss Function

MSE = (1/n) * Σ (yᵢ - ŷᵢ)²

3. Vectorized Gradients

dW = -2 * X.T @ (y - ŷ) / n
db = -2 * mean(y - ŷ)

4. Gradient Descent Step

W = W - lr * dW
b = b - lr * db

Where lr is the learning rate.

Code Example: One Step

import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([10, 20, 30])

W = np.zeros(X.shape[1])
b = 0.0
lr = 0.01

# Predict
y_pred = X @ W + b

# Compute loss
loss = np.mean((y - y_pred) ** 2)

# Compute gradients
dW = -2 * X.T @ (y - y_pred) / len(y)
db = -2 * np.mean(y - y_pred)

# Update
W = W - lr * dW
b = b - lr * db

Assignments

All implementations go in day05_assignments/:

assignment1_predict_multifeature.py→ def predict(X, W, b) -> np.ndarray

assignment2_mse_multifeature.py→ def mse_loss(y_true, y_pred) -> float

assignment3_compute_gradients.py→ def compute_gradients(X, y_true, y_pred) -> tuple[np.ndarray, float]

assignment4_gradient_step_vectorized.py→ def gradient_step(W, b, dW, db, learning_rate) -> tuple[np.ndarray, float]

assignment5_train_model.py→ def train(X, y, epochs, learning_rate) -> tuple[np.ndarray, float, list[float]]

Reflection

Using vectorized math makes ML both elegant and efficient. This prepares me for deep learning models where layers are just matrix operations under the hood.

Next Up: Day 06 – Statistics Refresher: Variance, Covariance, Correlation