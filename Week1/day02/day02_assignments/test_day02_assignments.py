import unittest
import numpy as np

from day02.day02_assigments.assignment1_array_creation import *
from day02.day02_assigments.assignment2_array_math import *
from day02.day02_assigments.assignment3_slicing_indexing import *
from day02.day02_assigments.assignment4_reshaping import *
from day02.day02_assigments.assignment5_broadcasting import *

class TestDay02Assignments(unittest.TestCase):

    def test_array_creation(self):
        self.assertTrue(np.array_equal(create_0_to_9(), np.arange(10)))
        self.assertTrue(np.array_equal(create_3x3_ones(), np.ones((3, 3))))
        self.assertTrue(np.array_equal(create_identity(), np.eye(3)))
        self.assertTrue(np.allclose(create_linspace(), np.linspace(0, 1, 5)))

    def test_array_math(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([5, 4, 3, 2, 1])
        add, mult, div, sq, mean_, std_ = array_math_ops(a, b)
        np.testing.assert_array_equal(add, np.array([6, 6, 6, 6, 6]))
        np.testing.assert_array_equal(mult, np.array([5, 8, 9, 8, 5]))
        np.testing.assert_array_equal(sq, np.array([1, 4, 9, 16, 25]))
        self.assertAlmostEqual(mean_, np.mean(add))
        self.assertAlmostEqual(std_, np.std(add))

    def test_slicing_indexing(self):
        mat = np.arange(16).reshape(4, 4)
        first_row, last_col, submatrix, reversed_mat = slice_and_index(mat)
        np.testing.assert_array_equal(first_row, mat[0])
        np.testing.assert_array_equal(last_col, mat[:, -1])
        np.testing.assert_array_equal(submatrix, mat[1:3, 1:4])
        np.testing.assert_array_equal(reversed_mat, mat[::-1, ::-1])

    def test_reshaping(self):
        arr = np.arange(16)
        reshaped, diagonal, flattened, raveled = reshape_and_flatten(arr)
        self.assertEqual(reshaped.shape, (4, 4))
        np.testing.assert_array_equal(diagonal, np.diag(reshaped))
        self.assertEqual(flattened.shape, (16,))
        self.assertEqual(raveled.shape, (16,))
        self.assertTrue(np.array_equal(flattened, raveled))

    def test_broadcasting(self):
        a = np.array([1, 2, 3])
        b = np.ones((3, 3))
        mult, row_add, col_sub = broadcast_ops(a, b)
        np.testing.assert_array_equal(mult, b * a)
        np.testing.assert_array_equal(row_add, b + a)
        np.testing.assert_array_equal(col_sub, b - a.reshape(3, 1))

if __name__ == "__main__":
    unittest.main()
