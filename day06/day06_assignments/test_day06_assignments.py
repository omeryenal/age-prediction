import unittest
import numpy as np

from assignment1_variance import calculate_variance
from assignment2_std import calculate_std
from assignment3_covariance import calculate_covariance
from assignment4_correlation import calculate_correlation
from assignment5_summary_stats import summarize_dataset

class TestDay06Assignments(unittest.TestCase):

    def test_variance(self):
        arr = np.array([1, 2, 3, 4, 5])
        expected = np.var(arr)
        result = calculate_variance(arr)
        self.assertAlmostEqual(result, expected)

    def test_std(self):
        arr = np.array([1, 2, 3, 4, 5])
        expected = np.std(arr)
        result = calculate_std(arr)
        self.assertAlmostEqual(result, expected)

    def test_covariance(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        expected = np.mean((x - np.mean(x)) * (y - np.mean(y)))
        result = calculate_covariance(x, y)
        self.assertAlmostEqual(result, expected)

    def test_correlation(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        expected = np.corrcoef(x, y)[0, 1]
        result = calculate_correlation(x, y)
        self.assertAlmostEqual(result, expected)

    def test_summary_stats(self):
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        stats = summarize_dataset(X)
        np.testing.assert_array_almost_equal(stats["mean"], np.mean(X, axis=0))
        np.testing.assert_array_almost_equal(stats["variance"], np.var(X, axis=0))
        np.testing.assert_array_almost_equal(stats["std"], np.std(X, axis=0))

if __name__ == "__main__":
    unittest.main()
