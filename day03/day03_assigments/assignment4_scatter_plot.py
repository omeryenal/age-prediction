"""
Assignment 4 – Scatter Plot of Height vs Weight

Write a function `plot_scatter(heights, weights)` that:
- Takes two equal-length lists or arrays (heights and weights)
- Plots a scatter plot
- Sets x-axis label to "Height (cm)"
- Sets y-axis label to "Weight (kg)"
- Sets title to "Height vs Weight"

Parameters:
- heights: list or np.ndarray
- weights: list or np.ndarray

Return:
- A matplotlib `Figure` object

Function Signature:
def plot_scatter(heights, weights) -> matplotlib.figure.Figure
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(heights, weights):
    fig, ax = plt.subplots()
    x = heights
    y = weights

    plt.scatter(x, y, color='green',alpha=0.7)
    plt.title("Scatter Plot of Height vs Weight")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.grid(True)

    plt.show()
    return fig

#heights = np.random.randint(10, 100, size=50)
#weights = np.random.randint(10, 100, size=50)
#plot_scatter(heights,weights)