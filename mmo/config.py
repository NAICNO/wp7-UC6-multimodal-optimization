#!/usr/bin/env python

# std libs
import warnings
import importlib

import numpy as np
import numpy.linalg as la

###############################################################################
# class
###############################################################################
class Config:
    def __init__(self, dim = -1, profile = 'standard', verbose = False):
        assert(dim > 0)
        assert(profile is not None)

        self.dim = dim
        self.dof_quad_polynomial = 1 + (3 * self.dim + self.dim * self.dim) // 2
        self.profile = profile
        self.verbose = verbose

        if self.profile == 'standard':
            self.n_pop = 10 * self.dof_quad_polynomial
            self.n_gen = 20
            self.seed_n_nb1 = self.dof_quad_polynomial - 1
            self.nb = None
            self.keep_old_pop = True
            self.deterministic = 1.0
            self.quadratic_fit = False
            self.cma_popsize = None

        else:
            assert(0)

        if self.verbose:
            print(self)

    def __str__(self):
        s = "config\n"
        s += f'  profile: {self.profile}\n'
        s += f'  dim: {self.dim}\n'
        s += f'  n_pop: {self.n_pop}\n'
        s += f'  keep_old_pop: {self.keep_old_pop}\n'
        s += f'  n_gen: {self.n_gen}\n'
        s += f'  seed_n_nb1: {self.seed_n_nb1}\n'
        if self.quadratic_fit:
            s += f'  seed_n_nb2: {self.seed_n_nb2}\n'
        else:
            if self.nb is None:
                s += f'  seed_n_nb2: dynamic\n'
            else:
                s += f'  seed_n_nb2: {self.nb.shape[0]}\n'
        s += f'  deterministic: {self.deterministic}\n'
        s += f'  quadratic fit: {self.quadratic_fit}\n'
        if self.cma_popsize is None:
            s += f'  cma_popsize: default\n'
        else:
            s += f'  cma_popsize: {self.cma_popsize}\n'
        return(s)




