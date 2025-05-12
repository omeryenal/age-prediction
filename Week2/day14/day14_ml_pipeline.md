# Day 14 – Complete ML Pipeline from Scratch

## Goals for Today
- Build a full machine learning pipeline without sklearn models
- Understand each stage: loading, preprocessing, splitting, training, evaluation
- Implement everything using NumPy
- Avoid data leakage
- Get comfortable with structuring complete ML projects

---

## Lecture 1 – What is a Machine Learning Pipeline?

An ML pipeline is a **sequence of steps** to go from raw data to a trained, evaluated model:

1. Load data
2. Preprocess features
3. Split into train/test
4. Train model (from scratch)
5. Evaluate results
6. Organize clean code structure


---

## Lecture 2 – Data Loading and Validation

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
assert not df.isnull().any().any(), "Missing values found!"

X = df.drop("target", axis=1).values
y = df["target"].values
```

Always inspect your data:

Check for missing values
Validate ranges and types
Plot distributions

## Lecture 3 – Manual Feature Standardization

Instead of StandardScaler, we implement Z-score scaling ourselves:

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

❗ Only use training data to compute mean/std — never fit on full data.

## Lecture 4 – Manual Train/Test Split
```python
def split_dataset(X, y, test_ratio=0.2):
    n = X.shape[0]
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = split_dataset(X_scaled, y)
```
## Lecture 5 – Linear Regression (Closed-form Solution)
```python
def train_linear_regression(X, y):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta
```
## Lecture 6 – Prediction and Evaluation
```python
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ theta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

theta = train_linear_regression(X_train, y_train)
y_pred = predict(X_test, theta)
loss = mse(y_test, y_pred)
print("Test MSE:", loss)
```
You can also compute R², MAE, etc. if needed.

## Lecture 7 – Pipeline Organization and Structure

Create modular functions:
```python
def preprocess(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

def train(X, y): ...
def evaluate(y_true, y_pred): 

    X, y = load_data("data.csv")
    X = preprocess(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    theta = train(X_train, y_train)
    y_pred = predict(X_test, theta)
    evaluate(y_test, y_pred)
```

Reflection

Today I built a full ML pipeline using nothing but NumPy. No prebuilt models, no hidden magic — just math, code, and logic. This gave me total control and helped me deeply understand how training and evaluation work.