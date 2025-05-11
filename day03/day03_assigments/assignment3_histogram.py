"""
Assignment 3 – Histogram of Normal Data

Write a function `plot_histogram(data)` that:
- Takes a 1D NumPy array of values (e.g. 1000 samples from np.random.normal)
- Plots a histogram with 30 bins
- Adds a vertical line showing the mean
- Adds title and axis labels

Parameters:
- data: np.ndarray of shape (n,)

Return:
- A matplotlib `Figure` object

Function Signature:
def plot_histogram(data: np.ndarray) -> matplotlib.figure.Figure
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data: np.ndarray):
    fig, ax = plt.subplots()
    mean_value = np.mean(data)
    plt.hist(data,bins=30,color='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_value:.2f}')
    plt.title("Histogram of Normal Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig

