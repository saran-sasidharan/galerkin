__author__ = 'numguy'

import numpy as np
import lobatto as lb
import integration as ig

n_lobatto = 5
element_order = 4

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


zeta = lb.root(n_lobatto)
weight = ig.lobatto_weight(n_lobatto)

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



res = element_diff_matrix(element_order, weight, zeta)
print element_diff_matrix(element_order, weight, zeta)
print np.sum(res)

