import numpy as np


def smape(y_pred, y):
    return np.mean(np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred)))
