# Day 08 – Cost Functions (MSE vs MAE)

## Goals for Today
- Understand the role of cost functions in model training
- Compare Mean Squared Error (MSE) and Mean Absolute Error (MAE)
- Learn how each reacts to outliers
- Implement both functions manually
- Visualize their error landscapes

---

## Concept Explanation

### 1. What is a Cost Function?

A cost (or loss) function measures how wrong a model's predictions are. During training, the model tries to minimize this value by adjusting its parameters.

---

### 2. Mean Squared Error (MSE)

MSE = (1/n) * Σ (yᵢ - ŷᵢ)²


- Penalizes larger errors more heavily due to squaring
- Sensitive to outliers

---

### 3. Mean Absolute Error (MAE)

MAE = (1/n) * Σ |yᵢ - ŷᵢ|

- Penalizes all errors linearly
- More robust to outliers than MSE

---

### 4. Comparison Example

```python
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred_good = np.array([2.5, 0.0, 2, 8])
y_pred_bad = np.array([0, 0, 0, 0])

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

print("Good prediction MSE:", mse(y_true, y_pred_good))  # small
print("Bad prediction MSE:", mse(y_true, y_pred_bad))    # large

print("Good prediction MAE:", mae(y_true, y_pred_good))  # small
print("Bad prediction MAE:", mae(y_true, y_pred_bad))    # still large but less extreme
```
---
### 5. When to Use MSE vs MAE?
Situation	Use
Want to heavily penalize large errors	MSE
Robustness to outliers	MAE
Optimization using gradients	MSE (smooth derivative)
Sparse data or median-like prediction	MAE
Assignments

Implement the following in day08_assignments/:

mse_loss(y_true, y_pred) -> float
mae_loss(y_true, y_pred) -> float
compare_loss(y_true, y_pred1, y_pred2) -> dict – returns both MSE and MAE for two predictions
plot_error_landscape(true_y) – plots MSE & MAE vs prediction
outlier_impact_analysis() – shows how outliers affect MSE and MAE
Reflection
---
- Today I learned that not all cost functions behave the same way. MSE is great for punishing large errors, - but that can be dangerous with outliers. MAE is gentler and often a better default for real-world, messy data.

### Next Up: Day 09 – Overfitting, Train/Test Split