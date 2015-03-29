__author__ = 'numguy'

import numpy as np
import lobatto as lb
import integration as ig
import matplotlib.pyplot as plt
from time import time


def lagrange(x_vector, x_find, x_index):
    '''
    i != j
    :param x_vector:
    :param x_find:
    :param x_index:
    :return:
    '''
    length = np.size(x_vector)
    prod = 1
    for i in range(length):
        if i != x_index:
            prod = prod * ((x_find-x_vector[i])/(x_vector[x_index]-x_vector[i]))
    return prod


def diff_lagrange(x_vector, x_find, x_index):
    length = np.size(x_vector)
    diff = 0
    denom  = 1
    for i in range(length):
        numer = 1
        if i != x_index:
            denom = denom * (x_vector[x_index]-x_vector[i])
            for j in range(length):
                if (j != i) and (j != x_index):
                    numer = numer * (x_find-x_vector[j])
            diff = diff + numer
    return diff/denom



def element_mass_matrix(element_order, weight, zeta):
    n = element_order+1
    x = lb.root(n)
    mass = np.zeros((n, n))
    n_weight = np.size(weight)
    psi_combinations = np.zeros((n_weight, n))
    for j in range(n_weight):
        for i in range(n):
            psi_combinations[j, i] = lagrange(x, zeta[j], i)
    for k in range(n_weight):
        temp_mass = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                temp_mass[j, i] = weight[k]*psi_combinations[k, i]*psi_combinations[k, j]
        mass = temp_mass + mass
    return mass/2


def element_diff_matrix(element_order, weight, zeta):
    n = element_order+1
    x = lb.root(n)
    diff = np.zeros((n, n))
    n_weight = np.size(weight)
    psi_combinations = np.zeros((n_weight, n))
    psi_diff_combinations = np.zeros((n_weight, n))
    for j in range(n_weight):
        for i in range(n):
            psi_combinations[j, i] = lagrange(x, zeta[j], i)
            psi_diff_combinations[j, i] = diff_lagrange(x, zeta[j], i)
    for k in range(n_weight):
        temp_diff = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                temp_diff[j, i] = weight[k]*psi_combinations[k, j]*psi_diff_combinations[k, i]
        diff = temp_diff + diff
    return diff


def rusinov_1d(element_order):
    row = element_order+1
    col = row+1
    rusin = np.zeros((row, col))
    rusin[0, 0] = -1.0
    rusin[-1, -1] = +1.0
    return rusin


def mass_stiff(n_element, element_mat):
    n, _ = np.shape(element_mat)
    N = ((n-1)*n_element)+1
    mass_mat = np.zeros((n_element, N, N))
    for i in range(n_element):
        k = i*(n-1)
        mass_mat[i, k:k+n, k:k+n] = element_mat
    stif_mat = np.sum(mass_mat, axis=0)
    k = (n_element-1)*(n-1)
    stif_mat[0, k:k+n-1] += stif_mat[N-1, k:k+n-1]
    stif_mat[N-1, k:k+n-1] = 0
    stif_mat[k:k+n-1, 0] += stif_mat[k:k+n-1, N-1]
    stif_mat[k:k+n-1, N-1] = 0
    return stif_mat


def diff_stiff(n_element, element_mat):
    n, _ = np.shape(element_mat)
    N = ((n-1)*n_element)+1
    diff_mat = mass_stiff(n_element, element_mat)
    diff_mat[N-1, N-1] = 0
    diff_mat[0, 0] = 0
    return diff_mat


def mass_stiff_disc(n_element, element_mat):
    n, _ = np.shape(element_mat)
    N = n*n_element
    stif_mat = np.zeros(( N, N))
    for i in range(n_element):
        k = i*n
        stif_mat[k:k+n, k:k+n] = element_mat
    return stif_mat


def diff_stiff_disc(n_element, element_mat):
    n, _ = np.shape(element_mat)
    rus = rusinov_1d(n-1)
    N = n*n_element
    diff_mat = mass_stiff_disc(n_element, element_mat)
    for i in range(1, n_element):
        k = i*n
        diff_mat[k:k+n, k-1:k+n] -= rus
    diff_mat[0, -1] -= rus[0, 0]
    diff_mat[n-1, n-1] -= rus[-1, -1]
    return diff_mat


def r_matrix(msm, dsm, u):
    R_mat = np.dot(np.linalg.inv(msm), dsm)*-1*u
    return R_mat


def x_domain(element_order, n_element, x_min=-1.0, x_max=+1.0):
    N = (element_order*n_element)+1
    x = np.zeros(N)
    Dx = abs(x_max-x_min)/n_element
    zeta = lb.root(element_order+1)
    for i in range(n_element):
        k = i*(element_order)
        x[k:k+element_order] = x_min + (zeta[:-1]+1.0)*(Dx/2.0)
        x_min = x_min+Dx
    x[N-1] = x_max
    return x


def x_domain_disc(element_order, n_element, x_min=-1.0, x_max=+1.0):
    N = (element_order+1)*n_element
    x = np.zeros(N)
    Dx = abs(x_max-x_min)/n_element
    zeta = lb.root(element_order+1)
    for i in range(n_element):
        k = i*(element_order+1)
        x[k:k+element_order+1] = x_min + (zeta+1.0)*(Dx/2.0)
        x_min = x_min+Dx
    return x


def dt_calc(courant, u, dx):
    return (courant*dx)/u


def rk_2nd(r_mat, initial, T, dt):
    steps = np.round(T/dt)
    steps.astype(int)
    q = initial
    for i in np.arange(steps):
        q_half = q + np.dot(r_mat, q)*(dt/2)
        q_temp = q + np.dot(r_mat, q_half)*dt
        q = q_temp
    return q


def l2_error(q_numer, q_exact):
    err = np.sqrt(np.mean((q_numer-q_exact)**2)/np.sum(q_exact**2))
    return err


def cont_galerkin(T, n_lobatto, element_order, n_element, u=2.0, courant=1.0/4.0, sigma=1.0/8.0):
    X = x_domain(element_order, n_element)
    initial = np.e**(-1*(X/(2*sigma))**2)
    Dx = np.min(abs(X[1:element_order+1]-X[0:element_order]))
    zeta = lb.root(n_lobatto)
    weight = ig.lobatto_weight(n_lobatto)
    t1 = time()
    edm = element_diff_matrix(element_order, weight, zeta)
    emm = element_mass_matrix(element_order, weight, zeta)*(2.0/n_element)
    dsm = diff_stiff(n_element, edm)
    msm = mass_stiff(n_element, emm)
    R_mat = r_matrix(msm, dsm, u)
    R_mat[abs(R_mat)<10**-10] = 0
    dt = dt_calc(courant, u, Dx)
    q = rk_2nd(R_mat, initial, T, dt)
    t2 = time()-t1
    return q, initial, X, t2


def disc_galerkin(T, n_lobatto, element_order, n_element, u=2.0, courant=1.0/4.0, sigma=1.0/8.0):
    X = x_domain_disc(element_order, n_element)
    initial = np.e**(-1*(X/(2*sigma))**2)
    Dx = np.min(abs(X[1:element_order+1]-X[0:element_order]))
    zeta = lb.root(n_lobatto)
    weight = ig.lobatto_weight(n_lobatto)
    t1 = time()
    edm = element_diff_matrix(element_order, weight, zeta)
    emm = element_mass_matrix(element_order, weight, zeta)*(2.0/n_element)
    edmD = np.transpose(edm)
    dsmD = diff_stiff_disc(n_element, edmD)
    msmD = mass_stiff_disc(n_element, emm)
    R_mat = r_matrix(msmD, dsmD, u)*-1
    R_mat[abs(R_mat)<10**-10] = 0
    dt = dt_calc(courant, u, Dx)
    q = rk_2nd(R_mat, initial, T, dt)
    t2 = time()-t1
    return q, initial, X, t2

#'''
T = 1.0
n_lobatto = 10
element_order = 8
n_element = 8
q, initial, X, _ = cont_galerkin(T, n_lobatto, element_order, n_element)
#q, initial, X, _ = disc_galerkin(T, n_lobatto, element_order, n_element)
err = l2_error(q, initial)
plt.plot(X, initial)
plt.plot(X, q, 'r*')
plt.legend(('Exact','Numer'))
plt.xlabel(r'$x$', fontsize='16')
plt.ylabel(r'$q(x,t)$', fontsize='16')
plt.show()
#'''

'''
# Exact CG problem
T = 1.0
flag = 2
element_order = np.array([1, 4, 8, 16])
n_element = np.array([16, 32, 64])
err = np.zeros(3)
for i in element_order:
    k = 0
    for j in n_element:
        q, initial, _, _ = disc_galerkin(T, i+flag, i, j/i)
        err[k] = l2_error(q, initial)
        print i, j, j/i, err[k]
        k += 1
    plt.plot(n_element+1, np.log10(err))
plt.legend(('order = 1', 'order = 4', 'order = 8', 'order = 16'))
plt.xlabel(r'$N_p$', fontsize='14')
plt.ylabel('L2 Error', fontsize='14')
plt.show()
'''

'''
T = 1.0
flag = 2
element_order = np.array([1, 4, 8, 16])
n_element = np.array([16, 32, 64])
err = np.zeros(3)
t_elap = np.zeros(3)
for i in element_order:
    k = 0
    for j in n_element:
        q, initial, _, t_elap[k] = disc_galerkin(T, i+flag, i, j/i)
        err[k] = l2_error(q, initial)
        print i, j, j/i, t_elap[k]
        k += 1
    plt.plot(t_elap, np.log10(err))
plt.legend(('order = 1', 'order = 4', 'order = 8', 'order = 16'))
plt.xlabel('Time (s)', fontsize='14')
plt.ylabel('L2 Error', fontsize='14')
plt.show()'''

'''
TIM = np.array([0.25, 0.5, 0.75, 1.0])
n_lobatto = 10
element_order = 8
n_element = 8
for T in TIM:
    #q, _, X, _ = cont_galerkin(T, n_lobatto, element_order, n_element)
    q, _, X, _ = disc_galerkin(T, n_lobatto, element_order, n_element)
    plt.plot(X, q)
plt.legend(('T = 0.25','T = 0.5','T = 0.75', 'T = 1.0'))
plt.xlabel(r'$x$', fontsize='16')
plt.ylabel(r'$q(x,t)$', fontsize='16')
plt.show()#'''