import numpy as np


def mape(y_pred, y):
    return np.sum(np.abs(y_pred - y)) / np.sum(y)
