# Day 12 – Feature Normalization & Scaling

## Goals for Today
- Understand why normalization/scaling is critical for ML
- Learn different methods: Min-Max, Z-score (Standardization), and Max-Abs
- Compare the effect of each method
- Apply scaling before training
- Avoid data leakage in test sets

---

## Lecture 1 – Why Scale Features?

Most ML models (especially gradient-based ones like linear regression, neural networks, and SVM) are sensitive to the **scale** of input features.

**Example problem:**
If one feature ranges from 1–1000 and another from 0–1, the model might:
- Prefer the large-scale feature
- Converge slowly
- Get stuck in bad local minima

➡️ Solution: Normalize or scale your data so that features are in comparable ranges.

---

## Lecture 2 – Min-Max Normalization

**Goal:** Rescale values to a 0–1 range.

**Formula:**
x_scaled = (x - min) / (max - min)


**Pros:**
- Keeps original distribution shape
- Good for bounded inputs (e.g. images: 0–255 → 0–1)

**Cons:**
- Sensitive to outliers
- Doesn’t center data

---

## Lecture 3 – Z-score Standardization

**Also called:** StandardScaler, mean-normalization

**Formula:**
x_scaled = (x - mean) / std


**Result:**
- Mean = 0
- Std = 1

**Pros:**
- Good for gradient descent
- Not affected by scale of units

**Use case:** Most common and robust scaler for ML models.

---

## Lecture 4 – Scaling in Practice (with NumPy)

```python
import numpy as np

def min_max_scale(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def z_score_scale(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

X = np.array([[10, 1000],
              [15, 1100],
              [20, 900]])

print("Min-Max:\n", min_max_scale(X))
print("Z-score:\n", z_score_scale(X))
```
You can also use sklearn.preprocessing.StandardScaler or MinMaxScaler.

Lecture 5 – Train/Test Leakage & Scaling Strategy

⚠️ Warning: Never fit your scaler on the entire dataset before splitting.

Correct Workflow:
1)Split into train/test
2)Fit scaler on training set
3)Apply same transformation to test set
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Reflection

Today I learned that models behave very differently depending on how inputs are scaled. Normalization is not just a detail — it's essential. Z-score standardization is often the best starting point, but I’ll always choose based on context.