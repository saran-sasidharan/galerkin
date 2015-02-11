__author__ = 'numguy'

import numpy as np
import matplotlib.pyplot as plt
import legendre as lg


def generate(x, order):
    '''
    Generates lobatto polynomial value of specified order and returns it value at a specific point in [-1,1] domain
    :param x: a row of values in interval [-1,1], where the lobatto polynomial to be evaluated
    :param order: order of lobatto polynomial
    :return: return the lobatto polynomial value at x for order 'order'
    '''
    num = np.size(x)
    poly = np.zeros((order+2, num))
    poly[0, :] = 0
    poly[1, :] = (0)*(0-x*lg.generate(x, 0))
    for i in range(2, order+2):
        poly[i, :] = (i-1)*(lg.generate(x, i-2)-x*lg.generate(x, i-1))
    return poly[order, :]


def bisection(init, end, convergence, order):
    '''
    Bisection method specifically for lobatto polynomials to find root interval
    :param init: initial point
    :param end: final point
    :param convergence: convergence criteria
    :param order: order of lobatto polynomial
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
    Secant method specifically for lobatto polynomials to find root
    :param init: initial point
    :param end: final point
    :param convergence: convergence criteria
    :param order: order of lobatto polynomial
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
    :param order: order of lobatto polynomial
    :param sec_convergence: Convergence criteria for secant method
    :param bi_convergence: Convergence criteria for bisection method
    :return: root of the polynomial
    '''
    init, end = bisection(init, end, bi_convergence, order)
    root = secant(init, end, sec_convergence, order)
    return root


def root(order):
    '''
    :param order: Order of lobatto polynomial
    :return: Roots of a lobatto polynomial
    '''
    sec_convergence = 1e-8
    bi_convergence = 1e-2
    interval = np.array([-1.0+(1e-15), 0, 1.0-(1e-15)])
    if order==2: return np.array([-1.0, 1.0])
    if order==3: return np.array([-1.0, 0, 1.0])
    for i in range(4, order+1):
        roots = np.zeros((i-2))
        for j in range(i-2):
            roots[j] = biMethodRoot(interval[j], interval[j+1], i, sec_convergence, bi_convergence)
        interval = np.concatenate((np.array([-1.0+(1e-15)]), roots, np.array([1.0-(1e-15)])))
    return np.concatenate((np.array([-1.0]), roots, np.array([1.0])))


def testing():
    order = 20
    plt.scatter(root(order), np.zeros((order)))
    plt.plot(np.linspace(-1, 1, 2000), lg.generate(np.linspace(-1, 1, 2000), order))
    plt.plot(np.linspace(-1, 1, 1000), generate(np.linspace(-1, 1, 1000), order))
    plt.show()

