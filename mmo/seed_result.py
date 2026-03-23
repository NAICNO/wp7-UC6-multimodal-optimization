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

###############################################################################
# function class
###############################################################################
def SeedsExclude(seed_res, idx):
    assert(len(idx)==seed_res.n_seeds)
    if seed_res.spread is None:
        filtered_seed_res = SeedResult(seeds=seed_res.seeds[~idx], spread=None, info=seed_res.info)
    else:
        if seed_res.corr_matrices is None:
            filtered_seed_res = SeedResult(seeds=seed_res.seeds[~idx], spread=seed_res.spread[~idx], info=seed_res.info)
        else:
            filtered_seed_res = SeedResult(seeds=seed_res.seeds[~idx], spread=seed_res.spread[~idx], corr_matrices=seed_res.corr_matrices[~idx], info=seed_res.info)
    return(filtered_seed_res)

class SeedResult():
    def __init__(self, seeds=None, spread=None, corr_matrices=None, info=''):
        self.seeds = seeds
        self.n_seeds = seeds.shape[0] if seeds is not None else 0
        self.spread = spread
        self.corr_matrices = corr_matrices
        if self.spread is not None:
            assert(self.spread.shape[0]==self.n_seeds)
        if self.corr_matrices is not None:
            assert(self.corr_matrices.shape[0]==self.n_seeds)
        self.info = info

    def __eq__(self, n):
        return(self.n_seeds==n)

    def info_prepend(self, info=''):
        self.info = info + self.info

    def info_append(self, info=''):
        self.info += info

    def __str__(self):
        s = ''
        s += "## Seed results ##\n"
        s += "n_seeds: %d\n" % self.n_seeds
        s += "Seeds:\n"
        s += str(self.seeds)
        s += "\n"
        s += "Spread:\n"
        s += "  %s\n" % str(self.spread)
        if self.corr_matrices is not None:
            s += "Correlation matrices (sqrt of eigenvalues, normalized):\n"
            for k in range(self.corr_matrices.shape[0]):
                w, __ = la.eig(self.corr_matrices[k])
                s += "  %s\n" % str(np.sqrt(w))
        return(s)


