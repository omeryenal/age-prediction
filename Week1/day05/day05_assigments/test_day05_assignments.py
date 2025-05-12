import unittest
import numpy as np

from assignment1_predict_multifeature import predict
from assignment2_mse_multifeature import mse_loss
from assignment3_compute_gradients import compute_gradients
from assignment4_gradient_step_vectorized import gradient_step
from assignment5_train_model import train

class TestDay05Assignments(unittest.TestCase):

    def test_predict(self):
        X = np.array([[1, 2], [3, 4]])
        W = np.array([0.5, 1.0])
        b = 1.0
        y_pred = predict(X, W, b)
        expected = np.array([1*0.5 + 2*1.0 + 1.0, 3*0.5 + 4*1.0 + 1.0])  # [3.0, 7.5]
        np.testing.assert_array_almost_equal(y_pred, expected)

    def test_mse_loss(self):
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([3.0, 5.0])
        loss = mse_loss(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(loss, expected)

    def test_compute_gradients(self):
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([10, 20])
        y_pred = np.array([8, 18])
        dW, db = compute_gradients(X, y_true, y_pred)

        # Manual computation
        errors = y_true - y_pred  # [2, 2]
        expected_dW = -2 * X.T @ errors / 2  # shape (2,)
        expected_db = -2 * np.mean(errors)

        np.testing.assert_array_almost_equal(dW, expected_dW)
        self.assertAlmostEqual(db, expected_db)

    def test_gradient_step(self):
        W = np.array([1.0, 2.0])
        b = 0.5
        dW = np.array([0.1, -0.2])
        db = 0.3
        lr = 0.01
        new_W, new_b = gradient_step(W, b, dW, db, lr)

        np.testing.assert_array_almost_equal(new_W, W - lr * dW)
        self.assertAlmostEqual(new_b, b - lr * db)

    def test_train(self):
        # y = 2*x1 + 3*x2 + 1
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1


        final_W, final_b, loss_history = train(X, y, epochs=5000, learning_rate=0.001)

        self.assertEqual(final_W.shape, (2,))
        self.assertIsInstance(final_b, float)
        self.assertGreater(len(loss_history), 0)
        self.assertTrue(loss_history[0] > loss_history[-1])  # loss must decrease

        # Check that weights are approximately correct
        self.assertAlmostEqual(final_W[0], 2.0, places=1)
        self.assertAlmostEqual(final_W[1], 3.0, places=1)
        self.assertAlmostEqual(final_b, 1.0, places=1)

if __name__ == "__main__":
    unittest.main()
