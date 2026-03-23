#!/usr/bin/env python

# std libs
import warnings
import importlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon

import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as ssd

# own
from mmo.integrate import xy_add, xy_reduce_xequal_ymin
from mmo.verbose import Verbose 

###############################################################################
# class
###############################################################################
class SeedSolveCollect:
    def __init__(self, f=None, domain=None, localsolver=None, seed=None, solutions=None, n_iter=-1, n_sol=-1, budget=None, verbose=0, plot=False, balance=True):
        # basics
        assert(f.dim==domain.dim)
        assert(f.dim==solutions.dim)
        self.dim = f.dim
        self.f = f
        self.domain = domain
        self.log_X = np.zeros((1,2))
        self.balance = balance
        self.nb_scale = 0.5
        self.balancing_reached = True

        # main components
        self.localsolver = localsolver
        self.seed = seed
        self.solutions = solutions

        # config of the overall algorithm
        self.n_iter = n_iter
        self.budget = budget
        self.verbose = Verbose(verbose)
        self.plot = plot
        self.n_sol_to_find = n_sol

        self.n_fev_all = 0
        self.n_fev_seeds = 0
        self.n_fev_solves = 0

        self.n_fev_all_cum = 0
        self.n_fev_seeds_cum = 0
        self.n_fev_solves_cum = 0

        self.sol_in_range_cum = 0

    def solve(self):
        # iterate: get seeds, local solve, collect
        k = 0
        while(k!=self.n_iter):
            # get seeds
            n_fev_before_seed = self.f.n_fev
            seeds_res = self.seed()
            if self.balance:
                seeds_res_additional = self.seed(nb_scale=self.nb_scale)
            if self.f.n_fev <= self.budget:
                self.n_fev_seeds = self.f.n_fev - n_fev_before_seed
                stop_bc_budget = False
            else:
                self.n_fev_seeds = 0
                stop_bc_budget = True
            self.n_fev_seeds_cum += self.n_fev_seeds
            self.n_fev_all = self.n_fev_seeds
            self.n_fev_all_cum += self.n_fev_seeds

            # store statistics 
            self.solutions.store_previous()

            # handle seeds one by one
            self.n_fev_solves = 0
            self.sol_in_range = 0
            for kk in range(seeds_res.n_seeds):

                # get seeds
                seed = seeds_res.seeds[kk]
                spread = seeds_res.spread[kk] if seeds_res.spread is not None else None
                corr_matrix = seeds_res.corr_matrices[kk] if seeds_res.corr_matrices is not None else None

                # seed in range of known solutions
                sol_in_range = self.solutions.solution_in_range(seed, spread)
                
                # local solve
                if sol_in_range == False:
                    n_fev = self.f.n_fev
                    x,y = self.localsolver.solve(seed, spread, corr_matrix)
                    if self.f.n_fev <= self.budget:
                        self.n_fev_solves += self.f.n_fev - n_fev
                        self.n_fev_solves_cum += self.f.n_fev - n_fev
                        self.n_fev_all += self.f.n_fev - n_fev
                        self.n_fev_all_cum += self.f.n_fev - n_fev
                else:
                    self.sol_in_range += 1
                    self.sol_in_range_cum += 1

                # stop bc of budget?
                if self.f.n_fev > self.budget:
                    stop_bc_budget = True
                    break

                n_fev = self.f.n_fev

                # update solutions 
                self.solutions.add(x,y)

            # if balance, handle additional seeds
            self.balancing_tried = False
            if self.balance and self.n_fev_seeds_cum > self.n_fev_solves_cum:
                self.balancing_tried = True
                self.balancing_reached = False
                for kk in range(seeds_res_additional.n_seeds):

                    # get seeds
                    seed = seeds_res_additional.seeds[kk]
                    spread = seeds_res_additional.spread[kk] if seeds_res_additional.spread is not None else None
                    corr_matrix = seeds_res_additional.corr_matrices[kk] if seeds_res_additional.corr_matrices is not None else None

                    # seed in range of known solutions
                    sol_in_range = self.solutions.solution_in_range(seed, spread)
                
                    # local solve
                    if sol_in_range == False:
                        n_fev = self.f.n_fev
                        x,y = self.localsolver.solve(seed, spread, corr_matrix)
                        if self.f.n_fev <= self.budget:
                            self.n_fev_solves += self.f.n_fev - n_fev
                            self.n_fev_solves_cum += self.f.n_fev - n_fev
                            self.n_fev_all += self.f.n_fev - n_fev
                            self.n_fev_all_cum += self.f.n_fev - n_fev
                    else:
                        self.sol_in_range += 1
                        self.sol_in_range_cum += 1

                    # stop bc of budget?
                    if self.f.n_fev > self.budget:
                        stop_bc_budget = True
                        break
    
                    n_fev = self.f.n_fev

                    # update solutions 
                    self.solutions.add(x,y)

                    # if balanced, stop
                    self.balancing_used = k + 1
                    if self.n_fev_seeds_cum < self.n_fev_solves_cum:
                        self.balancing_reached = True
                        break

            if self.balancing_reached == True:
                self.nb_scale = 0.5
            else:
                self.nb_scale /= 2

            k += 1
            self.verbose(1,"##############################")
            self.verbose(1,"## iteration %.3d" % k)
            self.verbose(1,"##############################")
            if self.balance:
                self.verbose(1,"balance")
                self.verbose(1,"  tried: %r" % self.balancing_tried)
                if self.balancing_tried:
                    self.verbose(1,"  additional seeds: %d" % seeds_res_additional.n_seeds)
                    self.verbose(1,"  additional seeds used: %d" % self.balancing_used)
                    self.verbose(1,"  reached: %r" % self.balancing_reached)
                    self.verbose(1,"  nb_scale: %f" % self.nb_scale)
                self.verbose(1,"")
            if seeds_res.info:
                self.verbose(1,seeds_res.info)
                self.verbose(1,"")
            self.verbose(1,"sol")
            self.verbose(1,"  total: %d (%d)" % (self.solutions.n_sol, self.solutions.n_sol - self.solutions.n_sol_previous))
            self.verbose(1,"  duplicates: %d (%d)" % (self.solutions.n_duplicates, self.solutions.n_duplicates - self.solutions.n_duplicates_previous))
            self.verbose(1,"  skipped solves:  %d (%d)" % (self.sol_in_range_cum, self.sol_in_range))
            self.verbose(1,"")
            self.verbose(1,"fct eval")
            self.verbose(1,"  seeds: %d (%d)" % (self.n_fev_seeds_cum,self.n_fev_seeds))
            self.verbose(1,"  solves: %d (%d)" % (self.n_fev_solves_cum,self.n_fev_solves))
            self.verbose(1,"  all: %d (%d)" % (self.n_fev_all_cum,self.n_fev_all))
            self.verbose(1,"")
            self.verbose(2,"y")
            self.verbose(2,"  local %s" % (str(np.sort(self.solutions.sol[:,-1],axis=None))))
            self.verbose(2,"")

            # stop
            if self.n_sol_to_find>0 and self.n_sol_to_find <= self.solutions.n_sol:
                break
            if stop_bc_budget:
                break

        return(self.solutions.sol)


    def __str__(self):
        s = ''
        s += "## Seed Solve Collect ##\n"
        s += "dim: %d" % self.dim
        s += "n_iter: %d" % self.n_iter
        s += "xtol: %f" % self.xtol
        s += "n_sol: %d" % self.xy.shape[0]
        return(s)


