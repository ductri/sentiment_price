import numpy as np


class MeanMetrics:

    def __init__(self):
        self.figures = []

    def add(self, value):
        self.figures.append(value)

    def mean(self):
        return np.mean(self.figures)

    def median(self):
        return np.median(self.figures)

    def get_count(self):
        return len(self.figures)

    def get_sum(self):
        return np.sum(self.figures)

    def reset(self):
        self.figures = []
