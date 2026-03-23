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
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
import scipy.spatial.distance as ssd
from scipy.sparse.csgraph import connected_components

###############################################################################
# functions
###############################################################################
def xy_add(xy1, xy2):
    assert(xy1.shape[1]==xy2.shape[1])
    xy = np.vstack((xy1, xy2))
    return(xy)

def xy_reduce_xequal_ymin(xy, tol = None):
    dim = xy.shape[1] - 1
    n = xy.shape[0]
    if n == 0: return(xy)

    x = xy[:, :dim].reshape(n, dim)
    d = squareform(ssd.pdist(x))
    eq = 1 * (d < tol)
    eq = np.tril(eq, k = -1)
    eq = csr_matrix(eq)
    n_components, labels = connected_components(csgraph = eq, directed = False, return_labels = True)

    y = xy[:,dim]
    y_min = np.ones(n_components) * (1 + np.max(y))
    xy_reduced = np.zeros((n_components, xy.shape[1]))
    for k in range(n):
        cc = labels[k]
        if y[k] < y_min[cc]:
            xy_reduced[cc] = xy[k]
            y_min[cc] = y[k]

    return(xy_reduced)






