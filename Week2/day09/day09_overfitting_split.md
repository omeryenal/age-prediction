# Day 09 – Overfitting and Train/Test Split

## Goals for Today
- Understand what overfitting and underfitting are
- Learn why we split datasets into train/test sets
- Implement your own data splitting function
- Practice visualizing model performance on both sets
- Build intuition for generalization

---

## Concept Explanation

### 1. What is Overfitting?

Overfitting happens when a model learns the training data **too well**, including noise and outliers. It performs well on training data but poorly on unseen data.

- High training accuracy  
- Low test accuracy  
- Model is too complex

---

### 2. What is Underfitting?

Underfitting occurs when a model is too simple to capture the underlying pattern in the data.

- Low training accuracy  
- Low test accuracy  
- Model is too weak

---

### 3. Train/Test Split

To detect overfitting or underfitting, we divide our data:

Train set: used to fit the model
Test set: used to evaluate generalization


A common ratio is 80% train / 20% test. No data leakage from test into training!

---

### 4. Example: Overfit vs Generalized

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, size=100)

x_train = x[:80]
y_train = y[:80]
x_test = x[80:]
y_test = y[80:]

# Fit a 3rd-degree poly (generalized)
p1 = np.poly1d(np.polyfit(x_train, y_train, 3))
# Fit a 15th-degree poly (overfit)
p2 = np.poly1d(np.polyfit(x_train, y_train, 15))

plt.scatter(x_train, y_train, label="Train", alpha=0.5)
plt.plot(x, p1(x), label="Degree 3", color="green")
plt.plot(x, p2(x), label="Degree 15", color="red")
plt.legend()
plt.title("Overfitting vs Generalization")
plt.show()
```
## Assignments

- Create these in day09_assignments/:

split_dataset(X, y, test_ratio)
→ Shuffles and splits into train/test
polynomial_fit_and_eval(X_train, y_train, X_test, y_test, degree)
→ Fits polynomial, returns train & test MSE
plot_train_test_fit(X_train, y_train, X_test, y_test, degree)
→ Plots both fits on same graph
overfitting_curve(X, y, degrees)
→ For each degree, compute train/test loss, plot curves
underfit_overfit_demo()
→ Run 3 models: underfit, good fit, overfit and show results
## Reflection

Overfitting makes your model memorize instead of generalizing. Today I learned how to detect this by looking at both training and test performance. It’s not just about accuracy — it’s about performance on unseen data.

Next Up: Day 10 – Model Evaluation Metrics