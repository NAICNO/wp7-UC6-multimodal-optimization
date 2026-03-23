#!/usr/bin/env python

# std libs
import warnings
import importlib.util

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
import scipy.spatial.distance as ssd
from sklearn.kernel_ridge import KernelRidge
import itertools

###############################################################################
# function class
###############################################################################
class Domain:
    """Axis-parallel hyper-cuboid domain class, dimension >= 1"""

    def __init__(self, boundary = None, verbose = False):
        """Initialize hyper-cuboid domain.

        Create an axis-parallel domain of hyper-cuboidal shape by providing the 
        lower-left (LL) and the upper-right (UR) point of the cuboid.

        Parameters
        ----------
        boundary : list
            List of two points: LL, UR. Each point is given as a list of d coordinates, d dimension.
        verbose : bool
            display configuration if True
        """

        self.ll = np.array(boundary[0])
        self.ur = np.array(boundary[1])
        self.l = self.ur - self.ll
        assert(np.all(self.ur - self.ll) > 0)
        self.center = 0.5 * (self.ll + self.ur)
        self.dim = self.center.shape[0]
        self.diameter = la.norm(self.ur - self.ll)
        self.verbose = verbose

        if self.verbose:
            print(self)


    def project_into_domain(self, x):
        """Projects point to closest point on boundary wrt Euclidean distance."""

        x = x * np.heaviside(x-self.ll, 1) + self.ll * np.heaviside(self.ll-x, 0)
        x = x * np.heaviside(self.ur-x, 1) + self.ur * np.heaviside(x-self.ur, 0)
        return(x)

    def is_in(self, p=None):
        """Returns True if point p lies in domain or on boundary, else False"""

        c1 = np.all(self.ll<=p, axis=1)
        c2 = np.all(p<=self.ur, axis=1)
        return(np.logical_and(c1,c2))

    def __str__(self):
        """Return string representation of domain: its dimension, LL and UR corners and diameter."""

        s = 'domain\n'
        s += f'  dim: {self.dim}\n'
        s += f'  LL: {str(self.ll)}\n'
        s += f'  UR: {str(self.ur)}\n'
        s += f'  diameter: {self.diameter}\n'
        return(s)




