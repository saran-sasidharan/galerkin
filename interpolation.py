__author__ = 'numguy'

import numpy as np
import matplotlib.pyplot as plt
import legendre as lg
import lobatto as lb


def vandermode(x, order):
    '''
    Monomial vandermode matrix
    :param x: Points to be interpolated
    :param order: Order of interpolation
    :return: Vandermode matrix
    '''
    v = (np.dot(np.reshape(x, (np.size(x), 1)), np.ones((1, order+1))))**(np.arange(order+1))
    return v


def lagrange_x(x, x0, y0):
    '''
    Lagrange interpolation
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :param y0: Known value at x0
    :return: interpolated values at x
    L is the coefficients of lagrange interpolation polynomial
    '''
    num_nodal = np.size(x0)
    v = vandermode(x0, num_nodal-1)
    L = np.linalg.inv(v)
    coefficient = np.dot(L, y0)
    y = np.dot(vandermode(x, num_nodal-1), coefficient)
    return y


def lagrange_y(x, x0, y0):
    num_nodal = np.size(x0)
    num_points = np.size(x)
    y = np.zeros(num_points)
    for i in range(num_points):
        weight = np.ones((num_nodal))
        for j in range(num_nodal):
            for k in range(num_nodal):
                if j != k:
                    weight[j] = weight[j] * ((x[i]-x0[k])/(x0[j]-x0[k]))
            y[i] = y[i] + weight[j]*y0[j]
    return y


def lagrange(x, x0, y0):
    sample_points = np.size(x0)
    interp_points = np.size(x)
    y = np.zeros(interp_points)
    x_numer = np.zeros((sample_points, sample_points-1))
    x_common = np.zeros((sample_points, sample_points-1))
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
        y[i] = np.sum(L*y0)
    return y

def diff_lagrange(x, x0, y0):
    sample_points = np.size(x0)
    interp_points = np.size(x)
    y = np.zeros(interp_points)
    x_numer = np.zeros((sample_points, sample_points-1))
    x_common = np.zeros((sample_points, sample_points-1))
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
    return y



def function(x):
    '''
    :param x: function parameter
    :return: function value
    '''
    y = np.cos(np.pi*0.5*x)
    return y

def diff_function(x):
    '''
    :param x: function parameter
    :return: function value
    '''
    y = -1*np.pi*0.5*np.sin(np.pi*0.5*x)
    return y

def testing_equal():
    order = 64
    x0 = np.linspace(-1, 1, order+1)
    y0 = function(x0)
    x = np.linspace(-1, 1, 100)
    y = lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x, y)
    plt.plot(np.linspace(-1,1,100), function(np.linspace(-1,1,100)))
    plt.show()

def testing_legendre():
    order = 64
    x0 = lg.root(order+1)
    y0 = function(x0)
    x = np.linspace(x0[0], x0[-1], 100)
    y = lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x,y)
    plt.plot(np.linspace(x0[0], x0[-1], 100), function(np.linspace(x0[0], x0[-1], 100)))
    plt.show()

def testing_lobatto():
    order = 8
    x0 = lb.root(order+1)
    y0 = function(x0)
    x = np.linspace(x0[0],x0[-1],100)
    y = lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x,y)
    plt.plot(np.linspace(x0[0],x0[-1],100), function(np.linspace(x0[0],x0[-1],100)))
    plt.show()


def testing_diff_equal():
    order = 40
    x0 = np.linspace(-1, 1, order+1)
    y0 = function(x0)
    x = np.linspace(-1, 1, 100)
    y = diff_lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x, y)
    plt.plot(np.linspace(-1,1,100), diff_function(np.linspace(-1,1,100)))
    plt.show()

testing_diff_equal()