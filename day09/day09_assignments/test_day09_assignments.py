import unittest
import numpy as np

from assignment1_split_dataset import split_dataset
from assignment2_polynomial_fit_and_eval import polynomial_fit_and_eval
from assignment4_overfitting_curve import overfitting_curve
from assignment5_underfit_overfit_demo import underfit_overfit_demo

class TestDay09Assignments(unittest.TestCase):

    def test_split_dataset_shapes(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.2)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 50)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 50)
        self.assertEqual(X_train.shape[1], 2)

    def test_polynomial_fit_and_eval(self):
        X_train = np.linspace(0, 1, 10)
        y_train = 3 * X_train**2 + 2
        X_test = np.linspace(0, 1, 5)
        y_test = 3 * X_test**2 + 2
        train_mse, test_mse = polynomial_fit_and_eval(X_train, y_train, X_test, y_test, degree=2)
        self.assertLess(train_mse, 1e-3)
        self.assertLess(test_mse, 1e-3)

    def test_overfitting_curve_runs(self):
        X = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * X)
        try:
            overfitting_curve(X, y, degrees=[1, 3, 5])
        except Exception as e:
            self.fail(f"overfitting_curve raised an error: {e}")

    def test_underfit_overfit_demo_runs(self):
        try:
            underfit_overfit_demo()
        except Exception as e:
            self.fail(f"underfit_overfit_demo raised an error: {e}")

if __name__ == "__main__":
    unittest.main()
