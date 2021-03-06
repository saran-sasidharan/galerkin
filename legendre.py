__author__ = 'numguy'

import numpy as np


def generate(x, order):
    '''
    Generates Legendre polynomial value of specified order and returns it value at a specific point in [-1,1] domain
    :param x: a row of values in interval [-1,1], where the legendre polynomial to be evaluated
    :param order: order of legendre polynomial
    :return: return the legendre polynomial value at x for order 'order'
    '''
    num = np.size(x)
    poly = np.zeros((order+2, num))
    poly[0, :] = 1.0
    poly[1, :] = x
    for i in range(2, order+2):
        poly[i, :] = (((2.0*i-1.0)/i)*x*poly[i-1, :]) - (((i-1.0)/i)*poly[i-2, :])
    return poly[order, :]


def bisection(init, end, convergence, order):
    '''
    Bisection method specifically for legendre polynomials to find root interval
    :param init: initial point
    :param end: final point
    :param convergence: convergence criteria
    :param order: order of legendre polynomial
    :return: return the root interval [init, end]
    '''
    poly1 = generate(init, order)
    poly2 = generate(end, order)
    error = 1.0
    if poly1*poly2 > 0:
        return init, end
    while error > convergence:
        middle = (init+end)*0.5
        poly3 = generate(middle, order)
        if poly1*poly3 > 0:
            init = middle
        else:
            end = middle
        error = np.abs(generate((init+end)*0.5, order))
    return init, end


def secant(init, end, convergence, order):
    '''
    Secant method specifically for legendre polynomials to find root
    :param init: initial point
    :param end: final point
    :param convergence: convergence criteria
    :param order: order of legendre polynomial
    :return: return the root
    '''
    error = 1.0
    while error > convergence:
        poly1 = generate(init, order)
        poly2 = generate(end, order)
        slope = (poly2-poly1)/(end-init)
        end = init - poly1/slope
        poly2 = generate(end, order)
        error = np.abs(poly2)
    return end


def biMethodRoot(init, end, order, sec_convergence, bi_convergence):
    '''
    Both Bisection and secant are processed here to obtain a faster code
    :param init: initial point
    :param end: final point
    :param order: order of legendre polynomial
    :param sec_convergence: Convergence criteria for secant method
    :param bi_convergence: Convergence criteria for bisection method
    :return: root of the polynomial
    '''
    init, end = bisection(init, end, bi_convergence, order)
    root = secant(init, end, sec_convergence, order)
    return root


def root(order):
    '''
    :param order: Order of legendre polynomial
    :return: Roots of a legendre polynomial
    '''
    sec_convergence = 1e-14
    bi_convergence = 1e-3
    interval = np.array([-1.0, 0, 1.0])
    for i in range(2, order+1):
        roots = np.zeros((i))
        for j in range(i):
            roots[j] = biMethodRoot(interval[j], interval[j+1], i, sec_convergence, bi_convergence)
        interval = np.concatenate((np.array([-1.0]), roots, np.array([1.0])))
    return roots

