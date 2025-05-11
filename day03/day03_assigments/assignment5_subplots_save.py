"""
Assignment 5 – Subplots and Save Figure

Write a function `create_and_save_subplots(filename)` that:
- Creates a 1-row, 2-column subplot layout
- First plot: y = x line from x = 0 to 10
- Second plot: y = x² line from x = 0 to 10
- Sets appropriate titles
- Saves the figure to the given filename (e.g. "comparison_plot.png")
- Does NOT show the plot

Parameters:
- filename: str, path where the figure will be saved

Return:
- Nothing

Function Signature:
def create_and_save_subplots(filename: str) -> None
"""
import matplotlib.pyplot as plt
import numpy as np

def create_and_save_subplots(filename):
    
    x = np.linspace(0, 10, 100)
    y1 = x
    y2 = x**2
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x,y1,color='blue')
    axs[0].set_title("y=x")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True)

    axs[1].plot(x,y2,color='red')
    axs[1].set_title("y=x^2")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
