__author__ = 'numguy'

import numpy as np
import legendre as lg
import lobatto as lb


def legendre_weight(order):
    '''
    Legendre weight calculator
    :param order: Order of legendre polynomial
    :return: Legendre weights
    '''
    x = lg.root(order)
    weight = (2.0*(1-x**2))/((order*lg.generate(x, order-1))**2)
    return weight


def lobatto_weight(order):
    '''
    Lobatto weight calculator
    :param order: Order of legendre polynomial
    :return: Lobatto weights
    '''
    x0 = lb.root(order)
    weight = 2.0/(order*(order-1)*(lg.generate(x0, order-1))**2)
    return weight


def integrate(y, weights):
    '''
    :param y: Function value at integrating points
    :param weights: Weights at integrating points
    :return: Integrated value
    '''
    return np.sum(y*weights)

