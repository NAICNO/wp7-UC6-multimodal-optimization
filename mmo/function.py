#!/usr/bin/env python

# std libs
import warnings
import importlib

import numpy as np
import numpy.linalg as la

###############################################################################
# function class
###############################################################################
class Function:
    def __init__(self, f=None, domain=None, project=False):
        self.f = f.evaluate
        self.domain = domain
        self.dim = domain.dim
        self.n_fev = 0
        self.n_call = 0
        self._xyt = np.zeros((0, self.dim+2))
        self.n_tag = 0
        self.current_tag = -1
        self._record = 1
        self.project = project

    def tag(self):
        self.current_tag = self.n_tag
        self.n_tag += 1
        return(self.current_tag)

    def notag(self):
        self.current_tag = -1

    def record(self, rec):
        self._record = rec

    def __call__(self, x):
        if len(x.shape)==1:
            x = x.reshape(1,-1)
        n = x.shape[0]
        assert(x.shape[1] == self.dim)
        if n==0:
            return(np.zeros(0))
        if self.project and self.domain is not None:
            x = self.domain.project_into_domain(x)
        y = np.zeros(n)
        for k in range(n):
            y[k] = self.f(x[k])
            if self._record:
                self.n_call += 1
                self.n_fev += 1
                self._xyt = np.vstack((self._xyt, np.hstack((x[k],y[k],self.current_tag))))
        return(y)

    def xy(self):
        return(self._xyt[:,:self.dim+1])

    def x(self):
        return(self._xyt[:,:self.dim])

    def y(self):
        return(self._xyt[:,self.dim])

    def tagged_xy(self, tag):
        xyt = self._xyt[self._xyt[:,-1]==tag]
        return(xyt[:,:(self.dim+1)])

    def tagged_x(self, tag):
        xyt = self._xyt[self._xyt[:,-1]==tag]
        return(xyt[:,:self.dim])

    def tagged_y(self, tag):
        xyt = self._xyt[self._xyt[:,-1]==tag]
        return(xyt[:,self.dim])

    def xy_in_domain(self, domain=None):
        xy = self.xy()
        if xy.shape[0]>0:
            x = xy[:,:self.dim]
            idx_ll = np.all(x >= domain.ll, axis=1)
            idx_ur = np.all(domain.ur >= x, axis=1)
            idx = idx_ll & idx_ur
            xy = xy[idx]

        return(xy)

    def __str__(self):
        s = ''
        s += "## function ##\n"
        s += "dim: %d\n" % self.dim
        s += "fev: %d\n" % self.n_fev
        s += "call: %d\n" % self.n_call
        return(s)




