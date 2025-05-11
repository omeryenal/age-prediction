import unittest
import numpy as np

from assignment1_predict_function import predict
from assignment2_mse_loss import mse_loss
from assignment3_compute_gradients import compute_gradients
from assignment4_gradient_descent import gradient_descent_step
from assignment5_training_loop_plot import train_linear_model

class TestDay04Assignments(unittest.TestCase):

    def test_predict(self):
        x = np.array([1, 2, 3])
        W = 2.0
        b = 1.0
        expected = np.array([3, 5, 7])
        output = predict(x, W, b)
        np.testing.assert_array_equal(output, expected)

    def test_mse_loss(self):
        y_true = np.array([2, 4, 6])
        y_pred = np.array([3, 5, 7])
        expected_loss = np.mean((y_true - y_pred) ** 2)
        loss = mse_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, expected_loss)

    def test_compute_gradients(self):
        x = np.array([1, 2])
        y_true = np.array([2, 4])
        y_pred = np.array([3, 5])
        dW, db = compute_gradients(x, y_true, y_pred)

        expected_dW = -2 * np.mean(x * (y_true - y_pred))  # = -2 * mean([-1, -2]) = 3.0
        expected_db = -2 * np.mean(y_true - y_pred)        # = -2 * mean([-1, -1]) = 2.0

        self.assertAlmostEqual(dW, expected_dW)
        self.assertAlmostEqual(db, expected_db)

    def test_gradient_descent_step(self):
        W = 1.0
        b = 0.0
        dW = 0.5
        db = 1.0
        lr = 0.1
        W_new, b_new = gradient_descent_step(W, b, dW, db, lr)

        self.assertAlmostEqual(W_new, 0.95)
        self.assertAlmostEqual(b_new, -0.1)

    def test_train_linear_model(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([3, 5, 7, 9])  # True function: y = 2x + 1

        final_W, final_b, loss_history = train_linear_model(x, y, epochs=1000, learning_rate=0.01)

        # W and b should converge to ~2 and ~1
        self.assertAlmostEqual(final_W, 2.0, places=1)
        self.assertAlmostEqual(final_b, 1.0, places=1)

        # Loss should decrease over time
        self.assertTrue(loss_history[0] > loss_history[-1])

if __name__ == "__main__":
    unittest.main()
