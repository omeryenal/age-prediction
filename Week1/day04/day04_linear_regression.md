Day 04 – Linear Regression from Scratch

Goals for Today

Understand the core logic behind linear regression

Implement a model using only NumPy

Manually compute gradients and update weights

Train a simple model using gradient descent

Visualize training progress

Concept Explanation

1. The Linear Model (Hypothesis Function)

We model the relationship between input x and output y as:

ŷ = W * x + b

Where:

ŷ: predicted output

W: weight (slope)

b: bias (intercept)

2. Loss Function – Mean Squared Error (MSE)

MSE = (1/n) * Σ (yᵢ - ŷᵢ)²

This gives the average squared difference between predicted and true values.

3. Computing Gradients

∂MSE/∂W = -2 * mean(x * (y - ŷ))
∂MSE/∂b = -2 * mean(y - ŷ)

4. Gradient Descent Step

W = W - lr * ∂MSE/∂W
b = b - lr * ∂MSE/∂b

Where lr is the learning rate.

Code Example: One Step

import numpy as np

x = np.array([1, 2, 3])
y = np.array([3, 5, 7])

W = 0.0
b = 0.0
lr = 0.01

# Predict
y_pred = W * x + b

# Compute loss
loss = np.mean((y - y_pred)**2)

# Compute gradients
dW = -2 * np.mean(x * (y - y_pred))
db = -2 * np.mean(y - y_pred)

# Update weights
W = W - lr * dW
b = b - lr * db

Assignments

All implementations go in day04_assignments/:

assignment1_predict_function.py→ def predict(x, W, b) -> np.ndarray

assignment2_mse_loss.py→ def mse_loss(y_true, y_pred) -> float

assignment3_compute_gradients.py→ def compute_gradients(x, y_true, y_pred) -> tuple[float, float]

assignment4_gradient_descent.py→ def gradient_descent_step(W, b, dW, db, learning_rate) -> tuple[float, float]

assignment5_training_loop_plot.py→ def train_linear_model(x, y, epochs, learning_rate) -> tuple[float, float, list[float]]

Reflection

Understanding how weights change based on gradients gave me a solid foundation in optimization.Now I know what’s really happening behind .fit() in ML libraries.Training a model from scratch gave me confidence in loss, learning rate, and convergence mechanics.

Next Up: Day 05 – Multivariable Linear Regression

