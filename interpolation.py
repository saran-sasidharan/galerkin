__author__ = 'numguy'

import numpy as np

def lagrange(x, x0, y0):
    '''
    Lagrange interpolation
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :param y0: Known value at x0
    :return: y --> interpolated values at x
            lebes --> Lebesgue function values at x
    '''
    sample_points = np.size(x0)
    interp_points = np.size(x)
    y = np.zeros(interp_points)
    x_numer = np.zeros((sample_points, sample_points-1))
    x_common = np.zeros((sample_points, sample_points-1))
    lebes = np.zeros(interp_points)
    for i in range(sample_points):
        k = 0
        for j in range(sample_points):
            if i != j:
                x_common[i, k] = x0[j]
                k += 1
    x_denom = np.dot(np.reshape(x0, (sample_points, 1)), np.ones((1, sample_points-1)))
    for i in range(interp_points):
        x_numer *= 0
        x_numer += x[i]
        L_temp = (x_numer-x_common)/(x_denom-x_common)
        L = np.prod(L_temp, 1)
        lebes[i] = np.sum(abs(L))
        y[i] = np.sum(L*y0)
    return y, lebes

def diff_lagrange(x, x0, y0):
    '''
    Lagrange derivative interpolation
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :param y0: Known value at x0
    :return: y --> interpolated values at x
            lebes --> Lebesgue function values at x
    '''
    sample_points = np.size(x0)
    interp_points = np.size(x)
    y = np.zeros(interp_points)
    x_numer = np.zeros((sample_points, sample_points-1))
    x_common = np.zeros((sample_points, sample_points-1))
    lebes = np.zeros(interp_points)
    for i in range(sample_points):
        k = 0
        for j in range(sample_points):
            if i != j:
                x_common[i, k] = x0[j]
                k += 1
    x_denom = np.dot(np.reshape(x0, (sample_points, 1)), np.ones((1, sample_points-1)))
    for i in range(interp_points):
        x_numer *= 0
        x_numer += x[i]
        temp_num = (x_numer-x_common)
        temp_num_sum = np.zeros(sample_points)
        for j in range(sample_points-1):
            temp_num_sum += np.prod(np.delete(temp_num, j, 1), 1)
        L = temp_num_sum/np.prod((x_denom-x_common), 1)
        y[i] = np.sum(L*y0)
        lebes[i] = np.sum(abs(L))
    return y, lebes