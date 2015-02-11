__author__ = 'numguy'

import numpy as np
import matplotlib.pyplot as plt
import interpolation as ip
import legendre as lg
import lobatto as lb


def errorFunction(x, x0):
    '''
    Lebesgue interpolation error function
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :param y0: Known value at x0
    :return: interpolated error values at x
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
            y[i] = y[i] + abs(weight[j])
    return y

def constant(x, x0):
    '''
    :param x: Points to be interpolated
    :param x0: Points known in the domain
    :return: Lebesgue constant
    '''
    return np.max(errorFunction(x, x0))


def errorLone(x, y):
    y_actual = ip.function(x)
    L1 = np.sum(abs(y-y_actual))/np.sum(abs(y_actual))
    return L1


def errorLtwo(x, y):
    y_actual = ip.function(x)
    L2 = np.sqrt(np.sum((y-y_actual)**2))/np.sqrt(np.sum(y_actual**2))
    return L2


def testing_equal_error():
    order = 8
    x0 = np.linspace(-1, 1, order+1)
    y0 = ip.function(x0)
    x = np.linspace(-1,1,50)
    y = errorFunction(x, x0)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x0,np.zeros(order+1))
    plt.show()


def testing_legendre_error():
    order = 8
    x0 = lg.root(order+1)
    y0 = ip.function(x0)
    x = np.linspace(x0[0],x0[-1],50)
    y = errorFunction(x, x0)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x0,np.zeros(order+1))
    plt.show()


def testing_lobatto_error():
    order = 8
    x0 = lb.root(order+1)
    y0 = ip.function(x0)
    x = np.linspace(x0[0],x0[-1],50)
    y = errorFunction(x, x0)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(x0,np.zeros(order+1))
    plt.show()
