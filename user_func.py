__author__ = 'numguy'

import numpy as np
import matplotlib.pyplot as plt

def genLegendre(x,order):
    '''
    Generates Legendre polynomial value of specified order and returns it value at a specific point in [-1,1] domain
    :param x: a row of values in interval [-1,1], where the legendre polynomial to be evaluated
    :param order: order of legendre polynomial
    :return: return the legendre polynomial value at x for order 'order'
    '''
    num = np.size(x)
    poly = np.zeros((order+2,num))
    poly[0,:] = 1.0
    poly[1,:] = x
    for i in range(2,order+2):
        poly[i,:] = (((2.0*i-1.0)/i)*x*poly[i-1,:]) - (((i-1.0)/i)*poly[i-2,:])
    return poly[order,:]

def testing():
    plt.figure()
    plt.plot(genLegendre(np.linspace(-1,1,100),5))
    plt.show()

testing()