import unittest
import numpy as np

from assignment1_gradient_1d import gradient_descent_1d
from assignment3_gradient_2d import gradient_descent_2d
from assignment5_custom_loss_descent import descent_on_custom_loss

class TestDay07Assignments(unittest.TestCase):

    def test_gradient_descent_1d(self):
        history = gradient_descent_1d(start=0.0, lr=0.1, steps=50)
        self.assertIsInstance(history, list)
        self.assertTrue(len(history) == 50)
        # should converge to ~3
        self.assertAlmostEqual(history[-1], 3.0, places=1)

    def test_gradient_descent_2d(self):
        start = np.array([0.0, 0.0])
        history = gradient_descent_2d(start, lr=0.1, steps=100)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 100)
        final = history[-1]
        self.assertAlmostEqual(final[0], 2.0, places=1)
        self.assertAlmostEqual(final[1], -1.0, places=1)

    def test_custom_loss_descent(self):
        # f(w) = (w - 5)^2, grad = 2*(w - 5)
        loss_fn = lambda w: (w - 5)**2
        grad_fn = lambda w: 2 * (w - 5)
        history = descent_on_custom_loss(loss_fn, grad_fn, start=0.0, lr=0.1, steps=50)
        self.assertIsInstance(history, list)
        self.assertTrue(abs(history[-1] - 5.0) < 0.5)

if __name__ == "__main__":
    unittest.main()
