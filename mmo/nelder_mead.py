#!/usr/bin/env python

# std libs
import warnings
import importlib

def running_parallel():
    spec = importlib.util.find_spec("pyspark")
    return(spec is not None)

if not running_parallel():
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.patches import Polygon

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from scipy.signal import savgol_filter

###############################################################################
# function class
###############################################################################
class NelderMead():
    def __init__(self, f=None, domain=None, tol=1e-6, budget=None, verbose=0):
        assert(f.dim==domain.dim)
        self.dim = f.dim
        self.f = f
        self.domain = domain
        self.tol = tol
        self.n_fev = 0
        self.verbose = verbose
        self.xy_path_ = None

    def solve(self, seed, spread, corr_matrix):
        x = seed
        # Call minimize on the negative function so that maximizing f is equivalent to minimizing -f
        r = minimize(lambda x: -self.f(x), x, method='nelder-mead', 
                    options={'xatol': self.tol, 'fatol': self.tol, 'disp': self.verbose})
        # get solution path
        xy = self.f.xy()
        xy = xy[-r.nfev:,:]
        if xy.shape[0] >= 2 * self.dim + 1:
            self.xy_path_ = savgol_filter(xy, 2 * self.dim + 1, 1, axis=0)
        else:
            self.xy_path_ = xy

        # finish solve: update number of function evaluations
        self.n_fev += r.nfev

        success = r.success
        if success:
            if self.dim == 1:
                for k in range(r.x.shape[0]):
                    r.x[k] = max(r.x[k], self.domain.ll[0])
                    r.x[k] = min(r.x[k], self.domain.ur[0])
            else:
                if np.any(r.x < self.domain.ll): success = False
                if np.any(r.x > self.domain.ur): success = False

        # If minimization was successful, return the solution and the *maximized* value
        if success:
            # r.fun is now the minimum of -f(x), so the original maximum is -r.fun
            return (r.x, -r.fun)
        else:
            return (None, None)


    def xy_path(self):
        return(self.xy_path_)

    def __str__(self):
        s = ''
        s += "## Nelder Mead ##\n"
        s += "dim: %d\n" % self.dim
        s += "n_fev: %d\n" % self.n_fev
        return(s)


