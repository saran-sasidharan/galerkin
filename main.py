__author__ = 'numguy'
import numpy as np
import matplotlib.pyplot as plt
import legendre as lg
import lobatto as lb
import interpolation as ip
import integration as ig


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

order = 15

number_of_points = 100


x0_eq = np.linspace(-1, 1, order+1)
x0_lg = lg.root(order+1)
x0_lb = lb.root(order+1)

x_eq = np.linspace(x0_eq[0], x0_eq[-1], number_of_points)
x_lg = np.linspace(x0_lg[0], x0_lg[-1], number_of_points)
x_lb = np.linspace(x0_lb[0], x0_lb[-1], number_of_points)


y0_eq = function(x0_eq)
y0_lg = function(x0_lg)
y0_lb = function(x0_lb)


y_eq, l_eq = ip.lagrange(x_eq, x0_eq, y0_eq)
y_lg, l_lg = ip.lagrange(x_lg, x0_lg, y0_lg)
y_lb, l_lb = ip.lagrange(x_lb, x0_lb, y0_lb)


plt.figure(1)
plt.suptitle('Lagrange interpolation & Lebesgue function of equi-spaced sampling points', fontsize='18')
plt.subplot(121)
plt.plot(x_eq, y_eq, label='Interpolated')
plt.plot(x_eq, function(x_eq), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_eq, l_eq, label='Error')
plt.scatter(x0_eq, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(2)
plt.suptitle('Lagrange interpolation & Lebesgue function for sampling Legendre roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lg, y_lg, label='Interpolated')
plt.plot(x_lg, function(x_lg), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_lg, l_lg, label='Error')
plt.scatter(x0_lg, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(3)
plt.suptitle('Lagrange interpolation & Lebesgue function for sampling Lobatto roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lb, y_lb, label='Interpolated')
plt.plot(x_lb, function(x_lb), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_lb, l_lb, label='Error')
plt.scatter(x0_lb, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()

y_eq, l_eq = ip.diff_lagrange(x_eq, x0_eq, y0_eq)
y_lg, l_lg = ip.diff_lagrange(x_lg, x0_lg, y0_lg)
y_lb, l_lb = ip.diff_lagrange(x_lb, x0_lb, y0_lb)

plt.figure(4)
plt.suptitle('Lagrange derivative interpolation & Lebesgue function of equi-spaced sampling points', fontsize='18')
plt.subplot(121)
plt.plot(x_eq, y_eq, label='Interpolated')
plt.plot(x_eq, diff_function(x_eq), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_eq, l_eq, label='Error')
plt.scatter(x0_eq, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(5)
plt.suptitle('Lagrange derivative interpolation & Lebesgue function for sampling Legendre roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lg, y_lg, label='Interpolated')
plt.plot(x_lg, diff_function(x_lg), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_lg, l_lg, label='Error')
plt.scatter(x0_lg, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(6)
plt.suptitle('Lagrange derivative interpolation & Lebesgue function for sampling Lobatto roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lb, y_lb, label='Interpolated')
plt.plot(x_lb, diff_function(x_lb), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$f(x)$', fontsize='18')
plt.legend()
plt.subplot(122)
plt.plot(x_lb, l_lb, label='Error')
plt.scatter(x0_lb, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


x0_lg = lg.root(order)
x0_lb = lb.root(order)
y0_lg = function(x0_lg)
y0_lb = function(x0_lb)
weight_lg = ig.legendre_weight(order)
weight_lb = ig.lobatto_weight(order)
int_lg = ig.integrate(y0_lg, weight_lg)
int_lb = ig.integrate(y0_lb, weight_lb)
act = 4.0/np.pi

print 'Exact integral is ', act
print 'Legendre integral is ', int_lg
print 'Lobatto integral is ', int_lb

print 'DONE'
