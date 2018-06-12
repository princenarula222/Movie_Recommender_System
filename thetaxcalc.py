import numpy as np


def thetaxcalc(matrat, rated, nomtrain, nou, n, x, theta):
    alpha = 0.0016
    buf = np.subtract(np.matmul(theta.T, x).T, matrat)
    z = x
    for i in range(nomtrain):
        mul = np.ones((n, nou))
        w = np.multiply(buf[i], rated[i])
        mul = np.multiply(mul, w)
        x[:, i] = np.subtract(x[:, i], alpha*np.sum(np.multiply(mul, theta), axis=1).T)
    for j in range(nou):
        mul = np.ones((n, nomtrain))
        w = np.multiply(buf[:, j].T, rated[:, j])
        mul = np.multiply(mul, w)
        theta[:, j] = np.subtract(theta[:, j], alpha*np.sum(np.multiply(mul, z), axis=1).T)

    return x, theta
