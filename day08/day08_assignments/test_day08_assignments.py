import unittest
import numpy as np

from assignment1_mse_loss import mse_loss
from assignment2_mae_loss import mae_loss
from assignment3_compare_loss import compare_loss
from assignment4_plot_error_landscape import plot_error_landscape
from assignment5_outlier_impact import outlier_impact_analysis

class TestDay08Assignments(unittest.TestCase):

    def test_mse_loss(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        expected = np.mean((y_true - y_pred) ** 2)
        result = mse_loss(y_true, y_pred)
        self.assertAlmostEqual(result, expected)

    def test_mae_loss(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        expected = np.mean(np.abs(y_true - y_pred))
        result = mae_loss(y_true, y_pred)
        self.assertAlmostEqual(result, expected)

    def test_compare_loss(self):
        y_true = np.array([1, 2, 3])
        y_pred1 = np.array([1, 2, 4])
        y_pred2 = np.array([0, 0, 0])
        result = compare_loss(y_true, y_pred1, y_pred2)

        self.assertIn("mse1", result)
        self.assertIn("mae2", result)
        self.assertLess(result["mse1"], result["mse2"])
        self.assertLess(result["mae1"], result["mae2"])

    def test_plot_error_landscape(self):
        try:
            plot_error_landscape(np.array([2, 4, 6]))  # should not throw
        except Exception as e:
            self.fail(f"plot_error_landscape raised an exception: {e}")

    def test_outlier_impact_analysis(self):
        result = outlier_impact_analysis()
        self.assertIn("mse_with_outlier", result)
        self.assertGreater(result["mse_with_outlier"], result["mse_no_outlier"])
        self.assertGreaterEqual(result["mae_with_outlier"], result["mae_no_outlier"])

if __name__ == "__main__":
    unittest.main()
