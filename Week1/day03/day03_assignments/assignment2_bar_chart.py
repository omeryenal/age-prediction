"""
Assignment 2 – Bar Chart of Department Student Counts

Write a function `plot_departments()` that:
- Plots a bar chart for the following data:
    Departments = ["Engineering", "Business", "Arts", "Science"]
    Counts = [120, 90, 60, 100]
- Adds axis labels and a title

Return:
- A matplotlib `Figure` object

Function Signature:
def plot_departments() -> matplotlib.figure.Figure
"""
import matplotlib.pyplot as plt

def plot_departments():
    fig, ax = plt.subplots()
    x = ["Engineering", "Business", "Arts", "Science"]
    y = [120, 90, 60, 100]

    plt.bar(x,y)
    plt.title("Bar Chart of Department Student Counts")
    plt.xlabel("Department")
    plt.ylabel("Counts")
    plt.grid(axis="y")
    plt.show()
    return fig
