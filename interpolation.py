__author__ = 'numguy'

import numpy as np
import matplotlib.pyplot as plt
import legendre as lg
import lobatto as lb

def lagrange(x, x0, y0):
    '''
    Lagrange interpolation
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :param y0: Known value at x0
    :return: interpolated values at x
    '''
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


def function(x):
    '''
    :param x: function parameter
    :return: function value
    '''
    y = 1.0/(1.0+50.0*x**2)
    return y


def testing_equal():
    order = 4
    x0 = np.linspace(-1, 1, order+1)
    y0 = function(x0)
    x = np.linspace(-1,1,100)
    y = lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x,y)
    plt.plot(np.linspace(-1,1,100), function(np.linspace(-1,1,100)))
    plt.show()

def testing_legendre():
    order = 8
    x0 = lg.root(order+1)
    y0 = function(x0)
    x = np.linspace(x0[0],x0[-1],100)
    y = lagrange(x, x0, y0)
    plt.figure()
    plt.plot(x,y)
    plt.plot(np.linspace(x0[0],x0[-1],100), function(np.linspace(x0[0],x0[-1],100)))
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


testing_lobatto()



