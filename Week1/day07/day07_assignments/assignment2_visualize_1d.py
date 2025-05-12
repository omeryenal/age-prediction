"""
Assignment 2 – Visualize 1D Descent

Plot the function f(w) = (w - 3)^2 and the gradient descent path over it.
Use matplotlib for plotting.

Returns:
- None

Function Signature:
def plot_1d_descent() -> None
"""
import matplotlib.pyplot as plt
def gradient_descent_1d(start, lr, steps):
    history = []
    w = start
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)
        history.append(w)
    return history

def plot_1d_descent() -> None:
    def f(w):
        return (w - 3) ** 2

    def gradient_descent_1d(start, lr, steps):
        history = []
        w = start
        for _ in range(steps):
            w = w - lr * 2 * (w - 3)
            history.append(w)
        return history

    w_vals = [i * 0.1 for i in range(-20, 80)]
    f_vals = [f(w) for w in w_vals]

    history = gradient_descent_1d(start=0.0, lr=0.1, steps=10)
    descent_f_vals = [f(w) for w in history]

    plt.figure(figsize=(8, 5))
    plt.plot(w_vals, f_vals, label="f(w) = (w - 3)^2", color="blue")
    plt.scatter(history, descent_f_vals, color="red", label="Gradient descent steps")
    plt.plot(history, descent_f_vals, linestyle='--', color="red", alpha=0.5)

    for i, (w, y) in enumerate(zip(history, descent_f_vals)):
        plt.text(w, y + 0.3, f"{i}", fontsize=8, ha='center')

    plt.title("Gradient Descent on f(w) = (w - 3)^2")
    plt.xlabel("w")
    plt.ylabel("f(w)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
  
    
    plot_1d_descent()
