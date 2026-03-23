#!/usr/bin/env python

# std libs
import warnings
import importlib
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon
from pylab import get_current_fig_manager
from matplotlib.patches import Rectangle

import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
import scipy.spatial.distance as ssd
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import multivariate_normal

# own
from mmo.seed_result import SeedResult, SeedsExclude
from mmo.domain import Domain
from sklearn.neighbors import NearestNeighbors
import itertools
from mmo.ga_dc import GA_DC

###############################################################################
# basic functions RandomSeed
###############################################################################
def volume_of_orthotope(p1, p2):
    d = p1 - p2
    volume = np.abs(np.prod(d))
    return(volume)

def volume_of_orthotope_axis_1(p1, p2):
    d = p1 - p2
    volume = np.abs(np.prod(d, axis=1))
    return(volume)

def is_lower(p1, p2):
    return(np.all(p1<p2))

def is_lower_axis_1(p1, p2):
    return(np.all(p1<p2, axis=1))

def in_domain(domain=None, p=None):
    c1 = is_lower(domain.ll, p)
    c2 = is_lower(p, domain.ur)
    return(c1*c2)

def p_in_domain_distance_to_boundary(domain=None, p=None):
    p_in_domain = in_domain(domain=domain, p=p)
    assert(np.all(p_in_domain))
    d1 = np.min(np.abs(p - domain.ll), axis=1)
    d2 = np.min(np.abs(domain.ur - p), axis=1)
    d = np.min(np.vstack((d1,d2)), axis=0)
    return(d)

def distance_to_domain(domain=None, p=None):
    d1 = (domain.ll-p) * np.heaviside(domain.ll-p, 0)
    d1max = max(d1)
    d2 = (p-domain.ur) * np.heaviside(p-domain.ur, 0)
    d2max = max(d2)
    dmax = max(d1max,d2max)
    return(dmax)

###############################################################################
# Box
###############################################################################
class Seed:
    def __init__(self, f= None, domain=None, solutions=None, plot=0, verbose=0, config=None):
        assert(f.dim==domain.dim)
        self.dim = f.dim
        self.f = f
        self.domain = domain
        self.n_call = 0
        self.all_seeds = np.zeros((0, self.dim))
        self.last_seeds = np.zeros((0, self.dim))
        self.plot = plot
        self.solutions = solutions
        self.verbose = verbose
        self.initial_cum = 0
        self.exclude_bnd_cum = 0
        self.exclude_sol_cum = 0
        self.final_cum = 0
        self.config = config
        self.n_pop = config.n_pop
        self.n_gen = config.n_gen
        self.previous_pop = np.zeros((0,self.dim+1))

    def __call__(self, nb_scale=1):
        self.n_call += 1

        # get initial seeds and spreadings
        self.pop_x = (self.domain.ll + (self.domain.ur - self.domain.ll) * np.random.random((self.n_pop, self.dim))).reshape(-1,self.dim)
        self.pop_y = self.f(self.pop_x).reshape(-1,1)
        self.pop = np.hstack((self.pop_x, self.pop_y))
        if self.config.keep_old_pop:
            self.pop = np.vstack((self.pop, self.previous_pop))
        self.ga = GA_DC(f=self.f, domain=self.domain, pop=self.pop, verbose=self.verbose, config=self.config)
        #if self.n_call == 1:
            #self.convergence.add(population=self.pop)
        self.convergence = None
        self.ga.iter(self.n_gen, convergence=self.convergence)
        #print("mean", self.convergence.contraction())
        if self.config.keep_old_pop:
            self.previous_pop = self.ga.xy()

        seeds, spread, corr_matrices = self.ga.seeds(nb_scale=nb_scale)
        if seeds is None:
            result_1 = SeedResult()
        else:
            result_1 = SeedResult(seeds=seeds[:,:self.dim], spread=spread, corr_matrices=corr_matrices)

        if self.plot and self.n_call % self.plot == 0:
            self.ga.plot(out = "png/seeds_%.4d" % self.n_call)

        self.initial_cum += result_1.n_seeds
        info = 'seeds\n'
        info += '  population: %d\n' % self.pop.shape[0]
        info += '  initial: %d (%d)\n' % (self.initial_cum, result_1.n_seeds)
        result_1.info_prepend(info=info)

        # exclude seeds in the vicinity of known soluions
        if self.solutions.sol.shape[0] > 0 and result_1.n_seeds > 0 and result_1.seeds.shape[0]>0:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.solutions.sol[:, :self.dim])
            distances, indices = nbrs.kneighbors(result_1.seeds)
            distances = distances.reshape(-1)
            idx_keep = result_1.spread < distances
            n_excluded = result_1.seeds.shape[0] - sum(idx_keep)
            if result_1.corr_matrices is None:
                result_2 = SeedResult(seeds=result_1.seeds[idx_keep], spread=result_1.spread[idx_keep], info=result_1.info)
            else:
                result_2 = SeedResult(seeds=result_1.seeds[idx_keep], spread=result_1.spread[idx_keep], corr_matrices=result_1.corr_matrices[idx_keep], info=result_1.info)
        else:
            n_excluded = 0
            result_2 = result_1

        self.exclude_sol_cum += n_excluded
        result_2.info_append(info='  excluded (sol): %d (%d)\n' % (self.exclude_sol_cum, n_excluded))
        self.final_cum += result_2.n_seeds
        result_2.info_append(info='  final: %d (%d)' % (self.final_cum, result_2.n_seeds))

        # admin
        if result_2.n_seeds > 0:
            self.all_seeds = np.vstack((self.all_seeds, result_2.seeds))
        self.last_seeds = result_2.seeds
        self.solutions.seeds = result_2.seeds

        # plot
        if self.plot and self.n_call % self.plot == 0:
            self.solutions.plot(out = "png/sol_%.4d" % self.n_call, population=self.pop)

        return(result_2)

    def __str__(self):
        s = ''
        s += "## MaxSeed ##\n"
        s += "n_call: %d\n" % self.n_call
        return(s)




