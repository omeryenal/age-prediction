# Day 10 – Model Evaluation Metrics

## Goals for Today
- Learn how to measure the performance of classification models
- Understand what accuracy, precision, recall, and F1-score actually mean
- Explore confusion matrices in binary classification
- Understand when accuracy is misleading
- Prepare for fair evaluation of your AI models

---

## Lecture 1 – Accuracy: The Most Basic Metric

**Accuracy** is the ratio of correct predictions to total predictions:

accuracy = (TP + TN) / (TP + TN + FP + FN)


Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

**When it's useful:**
- Balanced datasets

**When it's misleading:**
- In imbalanced datasets (e.g., 95% of data is class 0), you can get 95% accuracy just by predicting 0 always.

---

## Lecture 2 – Precision and Recall

**Precision**: How many of your positive predictions were correct?

precision = TP / (TP + FP)

Think of it like this:
- **Precision** answers: “When I say yes, how often am I right?”
- **Recall** answers: “How many of the actual yeses did I find?”

**Use precision when** false positives are costly.  
**Use recall when** false negatives are costly.

---

## Lecture 3 – F1 Score: Balancing Precision and Recall

The **F1 Score** is the harmonic mean of precision and recall.

F1 = 2 * (precision * recall) / (precision + recall)


Why harmonic mean? Because it punishes imbalance more than the average.

**When to use it:**
- You care equally about precision and recall.
- You want a single number to represent both.

---

## Lecture 4 – Confusion Matrix: The Full Picture

A **confusion matrix** is a 2x2 table in binary classification:

        Predicted
         0     1
Actual 0 TN FP
       1 FN TP


It shows:
- What’s getting confused
- Exact counts for correct and incorrect predictions
- Basis for all other metrics

In multi-class problems, the matrix becomes NxN.

---

## Lecture 5 – Metric Selection: Which Should You Use?

| Metric     | Best When... |
|------------|--------------|
| Accuracy   | Classes are balanced |
| Precision  | False positives are costly (e.g. spam detection) |
| Recall     | False negatives are costly (e.g. cancer detection) |
| F1 Score   | Need balance in noisy or skewed data |
| Confusion Matrix | You want full transparency in evaluation |

**Key Idea:**  
No single metric is best. You must choose based on context.

---

## Reflection

Today I learned that model evaluation is not just about a single "score". Each metric tells a different story, and using the wrong one can lead to misleading conclusions. As I build more ML systems, I need to always ask: *“What are the consequences of being wrong?”*

---

## Next Up: Day 11 – Polynomial Regression
