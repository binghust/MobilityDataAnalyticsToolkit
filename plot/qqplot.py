# encoding=utf-8
from pickle import load

import matplotlib.pyplot as plt
import numpy as np


class QQPlot:
    """
    The quantile-quantile (q-q) plot is a graphical technique for determining
    if two data sets come from populations with a common distribution.
    Reference : https://www.itl.nist.gov/div898/handbook/eda/section3/qqplot.htm
    """

    def __init__(self, x, y, interval, sample_num):
        intervals = np.linspace(start=interval[0], stop=interval[1], num=sample_num + 1)
        intervals[0] -= np.finfo(float).eps
        self.quantiles_of_x = QQPlot.get_quantiles(x, intervals)
        self.quantiles_of_y = QQPlot.get_quantiles(y, intervals)

    @staticmethod
    def get_quantiles(values, intervals):
        assert values.min() > intervals[0]
        assert values.max() <= intervals[-1]
        interval_idx = 0
        quantiles = np.zeros(len(intervals) - 1)
        for idx, i in enumerate(sorted(values)):
            while i > intervals[interval_idx]:
                quantiles[interval_idx] = quantiles[interval_idx - 1]
                interval_idx += 1
            quantiles[interval_idx - 1] = idx + 1
        quantiles /= len(values)
        return quantiles

    def plot(self):
        plt.scatter(self.quantiles_of_x, self.quantiles_of_y)
        plt.show()


def qqplot_of_js_and_vfcs():
    with open('jaccard_similarities.dat', 'rb') as fh1, \
            open('actual_visit_frequency_cosine_similarities.dat', 'rb') as fh2:
        js = load(fh1)
        vfcs = load(fh2)
        qqplt = QQPlot(x=js.ravel(), y=vfcs.ravel(), interval=[0.0, 1.0], sample_num=1000)
        qqplt.plot()


def qqplot_of_avfcs_and_kvfcs():
    with open('actual_visit_frequency_cosine_similarities.dat', 'rb') as fh1, \
            open('5_anonymous_visit_frequency_cosine_similarities.dat', 'rb') as fh2:
        vfcs = load(fh1)
        kvfcs = load(fh2)
        qqplt = QQPlot(x=vfcs.ravel(), y=kvfcs.ravel(), interval=[0.0, 1.0], sample_num=1000)
        qqplt.plot()


if __name__ == '__main__':
    qqplot_of_js_and_vfcs()
    qqplot_of_avfcs_and_kvfcs()
