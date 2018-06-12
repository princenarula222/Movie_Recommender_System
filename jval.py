import numpy as np


def jval(x, theta, matrat, rated):
    summ = np.subtract(np.matmul(theta.T, x).T, matrat)
    summ = np.multiply(summ, summ)
    summ = np.multiply(summ, rated)
    summ = np.sum(summ, axis=None)
    return 0.5*summ
