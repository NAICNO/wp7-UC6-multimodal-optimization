#!/usr/bin/env python

# std libs
import warnings
import importlib

import numpy as np
import numpy.linalg as la

from mmo.cma import CMA
from mmo.domain import Domain
from mmo.function import Function

###############################################################################
# class
###############################################################################
class FitQuadraticPolynomial:
    def __init__(self, x, y):
        # admin
        self.x = x
        self.y = y
        self.dim = self.x.shape[1]
        self.dof = int(1 + self.dim + 0.5 * self.dim * (self.dim +1)) 
        self.n_eq = self.x.shape[0]
        self.xi_shape = (self.n_eq, int(0.5*self.dim*(self.dim+1)))
        self.eta_shape = (self.n_eq, self.dim)
        self.c_shape = (self.n_eq, 1)
        self.t_shape = (self.n_eq, self.dof)
        assert(self.n_eq >= self.dof)
        assert(self.n_eq == y.shape[0])

    def xi(self, g):
        assert(g.shape[0] == self.dof)
        al = g[(self.dim+1):]
        a = np.zeros((self.dim, self.dim))
        n = 0
        for k in range(self.dim):
            for kk in range(k, self.dim):
                a[k, kk] = al[n]
                a[kk, k] = al[n]
                n += 1
        return(a)

    def eta(self, g):
        assert(g.shape[0] == self.dof)
        b = g[1:(self.dim+1)]
        return(b)

    def c(self, g):
        assert(g.shape[0] == self.dof)
        return(g[0])

    def m(self):
        M = np.zeros(self.t_shape)
        for k in range(self.dof):
            g = np.eye(1, self.dof, k).reshape(-1)
            if k == 0:
                M[:,k] = self.c(g) * np.ones(self.n_eq)
            elif k < self.dim + 1:
                M[:,k] = np.dot(self.x, self.eta(g))
            else:
                M[:,k] = np.sum(self.x * np.dot(self.x, self.xi(g)), axis=1)
        return(M)

    def fit_abc(self):
        M = self.m()
        A = np.dot(np.transpose(M), M)
        b = np.dot(np.transpose(M), self.y).reshape(-1,1)
        g = la.solve(A,b)
        a = self.xi(g)
        b = self.eta(g)
        c = self.c(g)
        return(a, b, c)

def has_minimum(x, y):
    x0 = x[0,:]
    x = x - x0
    d0 = np.max(la.norm(x, axis=1))
    fit = FitQuadraticPolynomial(x, y)
    a, b, c = fit.fit_abc()
    e = la.eigvalsh(a)
    #print(e)
    x1 = - 0.5 * la.solve(a, b).reshape(-1)
    d1 = la.norm(x1)
    #print(d0, d1)
    if d0 >= d1 and np.all(e > 0):
        return(True)
    return(False)





