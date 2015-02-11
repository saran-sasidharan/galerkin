__author__ = 'numguy'
import numpy as np
import matplotlib.pyplot as plt
import legendre as lg
import lobatto as lb
import interpolation as ip
import lebesgue as lbs


'''
order = 3
number_of_points = 50


x0_eq = np.linspace(-1, 1, order+1)
x0_lg = lg.root(order+1)
x0_lb = lb.root(order+1)

x_eq = np.linspace(x0_eq[0], x0_eq[-1], number_of_points)
x_lg = np.linspace(x0_lg[0], x0_lg[-1], number_of_points)
x_lb = np.linspace(x0_lb[0], x0_lb[-1], number_of_points)

e_eq = np.zeros(63)
e_lg = np.zeros(63)
e_lb = np.zeros(63)


y0_eq = ip.function(x0_eq)
y0_lg = ip.function(x0_lg)
y0_lb = ip.function(x0_lb)


y_eq = ip.lagrange(x_eq, x0_eq, y0_eq)
y_lg = ip.lagrange(x_lg, x0_lg, y0_lg)
y_lb = ip.lagrange(x_lb, x0_lb, y0_lb)


plt.figure(1)
plt.suptitle('Lagrange interpolation & Lebesgue function of order 8 for equi-spaced sampling points', fontsize='18')
plt.subplot(121)
plt.plot(x_eq, y_eq, label='Interpolated')
plt.plot(x_eq, ip.function(x_eq), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$cos(\frac{\pi}{2}x)$', fontsize='18')
plt.axis([-1, 1, 0, 1])
plt.legend()
plt.subplot(122)
plt.plot(x_eq, lbs.errorFunction(x_eq, x0_eq), label='Error')
plt.scatter(x0_eq, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(2)
plt.suptitle('Lagrange interpolation & Lebesgue function of order 8 for sampling Legendre roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lg, y_lg, label='Interpolated')
plt.plot(x_lg, ip.function(x_lg), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$cos(\frac{\pi}{2}x)$', fontsize='18')
plt.axis([-1, 1, 0, 1])
plt.legend()
plt.subplot(122)
plt.plot(x_lg, lbs.errorFunction(x_lg, x0_lg), label='Error')
plt.scatter(x0_lg, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()


plt.figure(2)
plt.suptitle('Lagrange interpolation & Lebesgue function of order 8 for sampling Lobatto roots', fontsize='18')
plt.subplot(121)
plt.plot(x_lb, y_lb, label='Interpolated')
plt.plot(x_lb, ip.function(x_lb), label='Actual')
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$cos(\frac{\pi}{2}x)$', fontsize='18')
plt.axis([-1, 1, 0, 1])
plt.legend()
plt.subplot(122)
plt.plot(x_lb, lbs.errorFunction(x_lb, x0_lb), label='Error')
plt.scatter(x0_lb, np.zeros(order+1), label='Sampling points')
plt.legend()
plt.xlabel(r'$x$', fontsize='18')
plt.ylabel(r'$e(x)$', fontsize='18')
plt.xlim([-1, 1])
plt.show()
'''


number_of_points = 50

e_eq1 = np.zeros(63)
e_eq2 = np.zeros(63)

for i in range(63):
    print i
    x0_eq = np.linspace(-1, 1, i+3)
    x_eq = np.linspace(x0_eq[0], x0_eq[-1], number_of_points)
    y0_eq = ip.function(x0_eq)
    y_eq = ip.lagrange(x_eq, x0_eq, y0_eq)
    e_eq1[i] = lbs.errorLone(x_eq, y_eq)
    e_eq2[i] = lbs.errorLtwo(x_eq, y_eq)

plt.figure(1)
plt.title('Equi-spaced interpolation error against order', fontsize='18')
plt.plot(np.arange(2, 65), np.log10(e_eq1), label='L1')
plt.plot(np.arange(2, 65), np.log10(e_eq2), label='L2')
plt.legend()
plt.xlabel(r'$N$', fontsize='18')
plt.ylabel(r'$\log_{10} (error)$', fontsize='18')
plt.xlim([2, 70])
plt.show()


number_of_points = 50

e_lg1 = np.zeros(63)
e_lg2 = np.zeros(63)

for i in range(63):
    print i
    x0_lg = lg.root(i+2)
    x_lg = np.linspace(x0_lg[0], x0_lg[-1], number_of_points)
    y0_lg = ip.function(x0_lg)
    y_lg = ip.lagrange(x_lg, x0_lg, y0_lg)
    e_lg1[i] = lbs.errorLone(x_lg, y_lg)
    e_lg2[i] = lbs.errorLtwo(x_lg, y_lg)

plt.figure(2)
plt.title('Legendre root interpolation error against order', fontsize='18')
plt.plot(np.arange(2, 65), np.log10(e_lg1), label='L1')
plt.plot(np.arange(2, 65), np.log10(e_lg2), label='L2')
plt.legend()
plt.xlabel(r'$N$', fontsize='18')
plt.ylabel(r'$\log_{10} (error)$', fontsize='18')
plt.xlim([2, 70])
plt.show()


number_of_points = 50

e_lb1 = np.zeros(63)
e_lb2 = np.zeros(63)

for i in range(63):
    print i
    x0_lb = lb.root(i+2)
    x_lb = np.linspace(x0_lb[0], x0_lb[-1], number_of_points)
    y0_lb = ip.function(x0_lb)
    y_lb = ip.lagrange(x_lb, x0_lb, y0_lb)
    e_lb1[i] = lbs.errorLone(x_lb, y_lb)
    e_lb2[i] = lbs.errorLtwo(x_lb, y_lb)

plt.figure(3)
plt.title('Lobatto root interpolation error against order', fontsize='18')
plt.plot(np.arange(2, 65), np.log10(e_lb1), label='L1')
plt.plot(np.arange(2, 65), np.log10(e_lb2), label='L2')
plt.legend()
plt.xlabel(r'$N$', fontsize='18')
plt.ylabel(r'$\log_{10} (error)$', fontsize='18')
plt.xlim([2, 70])
plt.show()