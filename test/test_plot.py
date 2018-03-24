# encoding=utf-8
import unittest

import numpy as np
from plot.qqplot import QQPlot


class TestQQPlot(unittest.TestCase):
    def setUp(self):
        self.sample_num = 100
        self.intervals = np.linspace(start=0.0, stop=1.0, num=self.sample_num + 1, endpoint=True)
        self.intervals[0] -= np.finfo(float).eps

    def test_get_quantiles_with_x0_equals_to_intervals0(self):
        x = np.array([0.000, 0.013, 0.999])
        quantiles = QQPlot.get_quantiles(x, self.intervals)
        self.assertEqual(quantiles[0], 1 / 3)
        self.assertEqual(quantiles[1], 2 / 3)
        self.assertEqual(quantiles[-1], 1.0)

    def test_get_quantiles_with_x0_greater_than_intervals0(self):
        x = np.array([0.001, 0.013, 0.999])
        quantiles = QQPlot.get_quantiles(x, self.intervals)
        self.assertEqual(quantiles[0], 1 / 3)
        self.assertEqual(quantiles[1], 2 / 3)
        self.assertEqual(quantiles[-1], 1.0)

    def test_get_quantiles_with_x0_greater_than_intervals1(self):
        x = np.array([0.012, 0.013, 0.999])
        quantiles = QQPlot.get_quantiles(x, self.intervals)
        self.assertEqual(quantiles[1], 2 / 3)
        self.assertEqual(quantiles[-1], 1.0)

    def test_qqplot_with_two_uniform_distributions(self):
        x = np.random.uniform(low=0.0, high=1.0, size=1000)
        y = np.random.uniform(low=0.0, high=1.0, size=1000)
        qqplt = QQPlot(x=x, y=y, interval=[0.0, 1.0], sample_num=1000)
        qqplt.plot()


if __name__ == '__main__':
    unittest.main()
