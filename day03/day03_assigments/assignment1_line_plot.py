"""
Assignment 1 – Line Plot of Squares

Write a function `plot_squares()` that:
- Creates a line plot for x = [1, 2, ..., 10] and y = x²
- Sets title to "Line Plot of Squares"
- Labels x-axis as "Number", y-axis as "Square"
- Shows grid lines

Return:
- A matplotlib `Figure` object

Function Signature:
def plot_squares() -> matplotlib.figure.Figure
"""
import matplotlib.pyplot as plt

def plot_squares():
    fig, ax = plt.subplots()
    x = list(range(1,11))
    y = [i**2 for i in x]
    plt.plot(x,y,marker='o')
    plt.title("Line Plot of Squares")
    plt.xlabel("Number")
    plt.ylabel("Square")
    plt.grid(True)
    plt.show()
    return fig

