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


def lagrange(x, x0, y0):
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


def function(x):
    '''
    :param x: function parameter
    :return: function value
    '''
    y = np.cos(np.pi*0.5*x)
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




