"""
Assignment 5 – Full Training Loop with Visualization

Write a function `train_linear_model(x, y, epochs, learning_rate)` that:

1. Initializes W and b as 0.0
2. For the given number of epochs:
    - Predicts ŷ
    - Calculates loss
    - Computes gradients
    - Updates weights with gradient descent
    - Tracks loss

3. Optionally: plots loss vs epoch using matplotlib

Parameters:
- x: np.ndarray of shape (n,)
- y: np.ndarray of shape (n,)
- epochs: int
- learning_rate: float

Returns:
- final_W: float
- final_b: float
- loss_history: list of float

Function Signature:
def train_linear_model(x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> tuple[float, float, list[float]]
"""
import numpy as np
import matplotlib.pyplot as plt

from assignment4_gradient_descent import gradient_descent_step

def train_linear_model(x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> tuple[float, float, list[float]]:
    W = 0.0
    b = 0.0
    loss_history = []
    n = len(x)

    for epoch in range(epochs):
        y_pred = W * x + b

        # Mean Squared Error Loss
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)

        # Gradients
        dW = (2/n) * np.sum((y_pred - y) * x)
        db = (2/n) * np.sum(y_pred - y)

        # Update
        W, b = gradient_descent_step(W, b, dW, db, learning_rate)

    # Plot
    plt.plot(range(epochs), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid(True)
    plt.show()

    return W, b, loss_history
