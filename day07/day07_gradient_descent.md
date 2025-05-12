Concept Explanation

1. What is Gradient Descent?

Gradient Descent is an optimization algorithm that iteratively adjusts parameters to minimize a given loss function.

In one dimension:

w = w - lr * dL/dw

Where:

w: weight (parameter to optimize)

lr: learning rate (step size)

dL/dw: gradient of the loss w.r.t. the weight

The gradient tells us the slope — we update in the opposite direction to reduce the loss.

2. Visualizing Gradient Descent in 1D

For the function f(w) = (w - 3)^2, the minimum is at w = 3. We can simulate updates:

import numpy as np
import matplotlib.pyplot as plt

w = 0.0
lr = 0.1
history = []

for _ in range(20):
    grad = 2 * (w - 3)
    w = w - lr * grad
    history.append(w)

plt.plot(history)
plt.title("Gradient Descent Convergence")
plt.xlabel("Step")
plt.ylabel("w")
plt.grid(True)
plt.show()

3. Gradient Descent in 2D

Function: f(w1, w2) = (w1 - 2)^2 + (w2 + 1)^2

Update rules:

df/dw1 = 2 * (w1 - 2)
df/dw2 = 2 * (w2 + 1)

Repeat updates until convergence:

w = np.array([0.0, 0.0])
lr = 0.1

for _ in range(100):
    grad = np.array([2 * (w[0] - 2), 2 * (w[1] + 1)])
    w = w - lr * grad

Assignments (day07_assignments/)

assignment1_gradient_1d.py→ def gradient_descent_1d(start: float, lr: float, steps: int) -> list[float]

assignment2_visualize_1d.py→ def plot_1d_descent() – shows f(w) and descent path

assignment3_gradient_2d.py→ def gradient_descent_2d(start: np.ndarray, lr: float, steps: int) -> list[np.ndarray]

assignment4_contour_plot.py→ def plot_2d_contour_path(history: list[np.ndarray]) -> None

assignment5_custom_loss_descent.py→ def descent_on_custom_loss(loss_fn, grad_fn, start: float, lr: float, steps: int) -> list[float]

Reflection

Today I understood gradient descent not just algebraically, but visually. The idea of moving downhill step by step feels very intuitive when you plot it. This helped solidify how training works in all optimization-based ML models.

Next Up: Day 08 – Cost Functions (MSE vs MAE)

