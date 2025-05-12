import numpy as np
from assignment1_predict_multifeature import predict
from assignment2_mse_multifeature import mse_loss
from assignment3_compute_gradients import compute_gradients
from assignment4_gradient_step_vectorized import gradient_step

def train(X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> tuple[np.ndarray, float, list[float]]:
    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, W, b)
        loss = mse_loss(y, y_pred)
        loss_history.append(loss)

        dW, db = compute_gradients(X, y, y_pred)
        W, b = gradient_step(W, b, dW, db, learning_rate)

    return W, b, loss_history
