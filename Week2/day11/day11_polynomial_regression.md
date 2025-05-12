# Day 11 – Polynomial Regression

## Goals for Today
- Understand how polynomial regression works
- Learn the trade-off between underfitting and overfitting
- Implement and visualize polynomial regression
- Learn how to choose the right polynomial degree
- Prepare for using nonlinear models in ML

---

## Lecture 1 – What is Polynomial Regression?

Polynomial regression fits the data to a curve rather than a straight line.

Equation:
y = b0 + b1x + b2x² + b3x³ + ... + bnxⁿ


You can transform the input `x` into `[x, x², x³, ...]` and use linear regression on this transformed input.

➡️ It’s still a **linear model** in terms of the parameters — but nonlinear in x.

---

## Lecture 2 – From Linear to Nonlinear with Features

To apply polynomial regression:
1. Start with your original feature `x`
2. Generate polynomial features up to a desired degree:
```python
X = np.column_stack([x**i for i in range(1, degree+1)])
```
Fit using linear regression on X
This allows the model to capture curvature and inflection in the data.

## Lecture 3 – Visualizing Underfitting vs Overfitting

Low degree (e.g. 1): underfit – model too simple
High degree (e.g. 15+): overfit – model too flexible
Medium degree: balanced

plt.scatter(x, y)
plt.plot(x, fitted_curve, label=f"Degree {d}")

➡️ Visual inspection is a powerful way to select degree before moving to more formal evaluation.

## Lecture 4 – Choosing the Right Degree

Use validation data or cross-validation:

Try multiple degrees
Compute train and test errors for each
Plot and pick the one that minimizes test error
Often, a degree between 2–5 works well for most datasets with curved trends.

## Lecture 5 – Use Cases of Polynomial Regression

Use Case	Why Polynomial?
Modeling curved trends	Captures nonlinearity
Sensor readings over time	Often noisy + curved
Predicting age from shape/size	Smooth but complex patterns
⚠️ Beware of extrapolation: polynomial models can behave wildly outside the training range.

Reflection

Polynomial regression showed me that not all models need to be straight lines. We can still use the power of linear algebra, but with more expressive inputs. I now understand how model complexity grows with degree and how to visualize the trade-offs.
