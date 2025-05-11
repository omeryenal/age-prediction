import unittest
import os
import matplotlib.pyplot as plt
import numpy as np

from assignment1_line_plot import *
from assignment2_bar_chart import *
from assignment3_histogram import *
from assignment4_scatter_plot import *
from assignment5_subplots_save import *

class TestDay03Assignments(unittest.TestCase):

    def test_line_plot(self):
        fig = plot_squares()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_chart(self):
        fig = plot_departments()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_histogram(self):
        data = np.random.normal(0, 1, 1000)
        fig = plot_histogram(data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_plot(self):
        heights = [150, 160, 170, 180, 190]
        weights = [50, 60, 70, 80, 90]
        fig = plot_scatter(heights, weights)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_subplots_and_save(self):
        filename = "comparison_plot.png"
        create_and_save_subplots(filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()
